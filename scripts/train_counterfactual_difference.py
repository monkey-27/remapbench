"""Train DiffCauseNet with counterfactual difference losses."""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import build_model
from remapbench.counterfactual_difference import (
    CAUSES, CounterfactualTransitionDataset, CounterfactualTupleDataset, split_path,
)
from scripts.evaluate import compute_multilabel_metrics, get_device
from scripts.final_common import common_metadata, config_hash, load_config, pair_names, write_json


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    return value


def _dice_loss(pred, target, eps=1e-5):
    dims = (2, 3)
    inter = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def _mask_loss(out, batch):
    target = batch["primitive_masks"]
    bce = F.binary_cross_entropy(out["evidence_maps"], target)
    return bce + _dice_loss(out["evidence_maps"], target)


def _inactive_loss(out, target):
    inactive = (target <= 0.5).float()
    evidence = out["evidence_maps"].mean(dim=(2, 3))
    return (evidence * inactive).sum() / inactive.sum().clamp_min(1.0)


def _transition_loss(model, batch, cfg):
    out = model(before_grid=batch["before_grid"], after_grid=batch["after_grid"])
    label = F.binary_cross_entropy_with_logits(out["cause_logits"], batch["target_multihot"])
    masks = _mask_loss(out, batch)
    inactive = _inactive_loss(out, batch["target_multihot"])
    total = (
        cfg.get("label_weight", 1.0) * label
        + cfg.get("primitive_mask_weight", 0.0) * masks
        + cfg.get("inactive_weight", 0.0) * inactive
    )
    return total, {"label_bce": label, "primitive_mask_loss": masks, "inactive_evidence": inactive}, out


def _cosine_gap(vec_a, vec_b):
    return 1.0 - F.cosine_similarity(vec_a, vec_b, dim=1).mean()


def _tuple_loss(model, batch, cfg):
    names = ("base_to_A", "base_to_B", "A_to_AB", "base_to_AB")
    losses, outs, metrics = [], {}, {}
    for name in names:
        loss, parts, out = _transition_loss(model, batch[name], cfg)
        losses.append(loss)
        outs[name] = out
        for key, value in parts.items():
            metrics[key] = metrics.get(key, 0.0) + value
    for key in list(metrics):
        metrics[key] = metrics[key] / len(names)

    # Existing tuples provide only A_then_B. This gives the valid B context term:
    # E_B(base->B) should match E_B(A->AB).
    b_active = batch["base_to_B"]["target_multihot"] > 0.5
    context_terms, raw_gaps = [], []
    for k in range(4):
        active = b_active[:, k]
        if active.any():
            gap = _cosine_gap(
                outs["base_to_B"]["evidence_vectors"][active, k],
                outs["A_to_AB"]["evidence_vectors"][active, k],
            )
            context_terms.append(gap)
            raw_gaps.append(gap.detach())
    zero = outs["base_to_A"]["cause_logits"].sum() * 0.0
    context = torch.stack(context_terms).mean() if context_terms else zero

    union_terms = []
    for source_name in ("base_to_A", "base_to_B"):
        target = batch[source_name]["target_multihot"]
        for k in range(4):
            active = target[:, k] > 0.5
            if active.any():
                union_terms.append(_cosine_gap(
                    outs[source_name]["evidence_vectors"][active, k],
                    outs["base_to_AB"]["evidence_vectors"][active, k],
                ))
    union = torch.stack(union_terms).mean() if union_terms else zero
    total = torch.stack(losses).mean()
    total = total + cfg.get("context_difference_weight", 0.0) * context
    total = total + cfg.get("mixed_union_weight", 0.0) * union
    metrics.update({
        "context_difference_loss": context,
        "context_gap_B_in_A_context": torch.stack(raw_gaps).mean() if raw_gaps else zero,
        "mixed_union_loss": union,
    })
    return total, {k: float(v.detach()) for k, v in metrics.items()}


@torch.no_grad()
def _eval_exact(model, loader, device):
    preds, targets = [], []
    model.eval()
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(before_grid=batch["before_grid"], after_grid=batch["after_grid"])
        preds.append(torch.sigmoid(out["cause_logits"]).cpu().numpy())
        targets.append(batch["target_multihot"].cpu().numpy())
    return compute_multilabel_metrics(np.concatenate(preds), np.concatenate(targets))


def _run_epoch(model, loader, cfg, device, optimizer=None):
    model.train(optimizer is not None)
    totals, n = {}, 0
    context = torch.enable_grad() if optimizer else torch.no_grad()
    with context:
        for batch in loader:
            batch = _to_device(batch, device)
            loss, metrics = _tuple_loss(model, batch, cfg)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            batch_n = len(batch["tuple_id"])
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach()) * batch_n
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * batch_n
            n += batch_n
    return {key: value / max(1, n) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-tuples", type=int)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device(cfg.get("device", "auto"))
    model = build_model(
        "counterfactual_difference",
        hidden_channels=cfg.get("hidden_channels", 48),
        use_diff_channels=cfg.get("use_diff_channels", True),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 1e-4))
    train_ds = CounterfactualTupleDataset(split_path(cfg["data_dir"], cfg.get("tuple_train_split", "train_tuple_seen")), args.max_tuples)
    val_ds = CounterfactualTransitionDataset(split_path(cfg["data_dir"], cfg.get("val_split", "val_composed_seen")))
    train_loader = DataLoader(train_ds, batch_size=cfg.get("tuple_batch_size", 128), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch_size", 256))
    run_dir = os.path.join("results", cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))
    metadata = {**common_metadata(cfg, args.config), "config_hash": config_hash(cfg)}
    best_exact, best_epoch, best_loss = -1.0, None, float("inf")
    epochs = args.epochs if args.epochs is not None else cfg.get("epochs", 15)
    with open(os.path.join(run_dir, "train_log.jsonl"), "w") as log:
        for epoch in range(1, epochs + 1):
            started = time.time()
            train = _run_epoch(model, train_loader, cfg, device, optimizer)
            val = _eval_exact(model, val_loader, device)
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train.items()},
                   "val_exact": val["exact_match"], "val_macro_f1": val["macro_f1"],
                   "val_micro_f1": val["micro_f1"], "seconds": time.time() - started}
            log.write(json.dumps(row) + "\n")
            log.flush()
            checkpoint = {
                **metadata, "epoch": epoch, "model_state": model.state_dict(),
                "model_name": "counterfactual_difference",
                "report_model_name": cfg.get("report_model_name", "counterfactual_difference"),
                "config": cfg, "seed": seed, "fold_id": cfg.get("fold_id"),
                "heldout_composed_pair_id": cfg.get("heldout_composed_pair_id"),
                "val_metrics": row,
            }
            torch.save(checkpoint, os.path.join(run_dir, "last.pt"))
            if val["exact_match"] > best_exact or (val["exact_match"] == best_exact and train["loss"] < best_loss):
                best_exact, best_epoch, best_loss = val["exact_match"], epoch, train["loss"]
                torch.save(checkpoint, os.path.join(run_dir, "best_val_composed_seen_exact.pt"))
            print(f"Ep {epoch:02d} loss={train['loss']:.4f} val_exact={val['exact_match']:.3f} "
                  f"context={train.get('context_difference_loss', 0.0):.4f} {row['seconds']:.1f}s")
    write_json(os.path.join(run_dir, "metrics_val.json"), {
        **metadata, "status": "complete", "best_val_exact": best_exact,
        "best_val_epoch": best_epoch, "checkpoint_metric_name": cfg.get("checkpoint_metric", "val_exact"),
        "checkpoint_metric_value": best_exact,
    })
    write_json(os.path.join(run_dir, "manifest.json"), {
        **metadata, "status": "complete", "method": "counterfactual_difference",
        "loss_name": "context_difference_loss",
        "causes": CAUSES, "heldout_pair_names": pair_names(cfg["heldout_composed_pair_id"]),
        "checkpoint_paths": {"last": "last.pt", "best_val_composed_seen_exact": "best_val_composed_seen_exact.pt"},
    })


if __name__ == "__main__":
    main()
