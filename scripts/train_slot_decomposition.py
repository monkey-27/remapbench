"""Three-stage trainer for the residual-grounded slot decomposition pilot."""
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
from torch.utils.data import ConcatDataset, DataLoader, Subset

from models import build_model
from models.slot_decomposition import PATHWAYS, SlotDecomposition
from remapbench.data import RemapDataset, TupleRemapDataset
from scripts.evaluate import compute_multilabel_metrics, get_device
from scripts.final_common import common_metadata, config_hash, load_config, pair_names, write_json


def _to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _zero(out):
    return out["remap_logits"].sum() * 0.0


def _attention_diversity(attention):
    flat = F.normalize(attention.flatten(2), dim=2)
    similarity = torch.einsum("bkc,blc->bkl", flat, flat)
    mask = ~torch.eye(similarity.shape[1], dtype=torch.bool, device=similarity.device)
    return similarity[:, mask].mean()


def _inactive_loss(out, target):
    inactive = (target <= 0.5).float()
    residual_norm = out["slot_residuals"].square().mean(dim=(2, 3, 4))
    routing = out["routing_weights"]
    denom = inactive.sum().clamp_min(1.0)
    return ((residual_norm + routing.square()) * inactive).sum() / denom


def _direct_loss(model, batch, cfg, oracle_residual_routing=True):
    out = model(batch["x"])
    target = batch["target_multihot"]
    residual_target = SlotDecomposition.residual_target(batch)
    if oracle_residual_routing:
        residual_total = (
            target[:, :, None, None, None] * out["slot_residuals"]
        ).sum(dim=1)
    else:
        residual_total = out["residual_total"]
    parts = {
        "label": F.binary_cross_entropy_with_logits(out["remap_logits"], target),
        "global_residual": F.mse_loss(residual_total, residual_target),
        "inactive": _inactive_loss(out, target),
        "diversity": _attention_diversity(out["slot_attention"]),
        "future_after": F.mse_loss(out["future_after"], batch["future_after"]),
        "value_after": F.mse_loss(out["value_after"], batch["value_after"]),
    }
    total = (
        cfg["label_weight"] * parts["label"]
        + cfg["global_residual_weight"] * parts["global_residual"]
        + cfg["inactive_weight"] * parts["inactive"]
        + cfg["diversity_weight"] * parts["diversity"]
        + cfg.get("future_after_weight", 0.1) * parts["future_after"]
        + cfg.get("value_after_weight", 0.1) * parts["value_after"]
    )
    return total, parts, out


def _component_and_contrastive(out_a, out_b, out_ab, a, b):
    component_terms, contrastive_terms = [], []
    for donor_out, donor in ((out_a, a), (out_b, b)):
        for index in range(len(PATHWAYS)):
            active = donor["target_multihot"][:, index] > 0.5
            if not active.any():
                continue
            donor_target = SlotDecomposition.residual_target(donor)
            component_terms.append(F.mse_loss(
                out_ab["slot_residuals"][active, index], donor_target[active]))
            ab_slot = F.normalize(out_ab["slot_vectors"][active, index], dim=1)
            donor_slot = F.normalize(donor_out["slot_vectors"][active, index], dim=1)
            wrong_index = (index + 1) % len(PATHWAYS)
            wrong_slot = F.normalize(donor_out["slot_vectors"][active, wrong_index], dim=1)
            positive = (ab_slot * donor_slot).sum(dim=1)
            negative = (ab_slot * wrong_slot).sum(dim=1)
            contrastive_terms.append(F.relu(0.2 - positive + negative).mean())
    zero = _zero(out_ab)
    component = torch.stack(component_terms).mean() if component_terms else zero
    contrastive = torch.stack(contrastive_terms).mean() if contrastive_terms else zero
    return component, contrastive


def _tuple_loss(model, batch, cfg):
    a, b, ab = (_to_device(batch[key], next(model.parameters()).device) for key in ("A", "B", "AB"))
    loss_a, _, out_a = _direct_loss(model, a, cfg)
    loss_b, _, out_b = _direct_loss(model, b, cfg)
    loss_ab, _, out_ab = _direct_loss(model, ab, cfg)
    component, contrastive = _component_and_contrastive(out_a, out_b, out_ab, a, b)
    total = (
        (loss_a + loss_b + loss_ab) / 3.0
        + cfg["component_residual_weight"] * component
        + cfg["contrastive_weight"] * contrastive
    )
    return total, {
        "loss": float(total.detach()),
        "component_residual_mse": float(component.detach()),
        "contrastive_loss": float(contrastive.detach()),
    }


def _epoch_direct(model, loader, cfg, device, optimizer=None):
    model.train(optimizer is not None)
    total, n = 0.0, 0
    context = torch.enable_grad() if optimizer else torch.no_grad()
    with context:
        for batch in loader:
            batch = _to_device(batch, device)
            loss, _, _ = _direct_loss(model, batch, cfg)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += float(loss.detach()) * len(batch["x"])
            n += len(batch["x"])
    return total / n


def _epoch_tuple(model, loader, cfg, device, optimizer=None):
    model.train(optimizer is not None)
    totals, n = {}, 0
    context = torch.enable_grad() if optimizer else torch.no_grad()
    with context:
        for batch in loader:
            loss, metrics = _tuple_loss(model, batch, cfg)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            batch_n = len(batch["tuple_id"])
            n += batch_n
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * batch_n
    return {key: value / n for key, value in totals.items()}


@torch.no_grad()
def _seen_metrics(model, loader, device):
    model.eval()
    predictions, targets = [], []
    for batch in loader:
        out = model(batch["x"].to(device))
        predictions.append(torch.sigmoid(out["remap_logits"]).cpu().numpy())
        targets.append(batch["target_multihot"].numpy())
    return compute_multilabel_metrics(np.concatenate(predictions), np.concatenate(targets))


def _loader(data_dir, splits, batch_size, shuffle=False, max_samples=None):
    datasets = [RemapDataset(os.path.join(data_dir, f"{split}.npz")) for split in splits]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage1-epochs", type=int)
    parser.add_argument("--stage2-epochs", type=int)
    parser.add_argument("--stage3-epochs", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-tuples", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device(cfg.get("device", "auto"))
    model = build_model("slot_decomposition", shared_pooled=cfg.get("shared_pooled", False)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    data_dir, batch_size = cfg["data_dir"], cfg["batch_size"]
    single_loader = _loader(data_dir, ["train_single"], batch_size, True, args.max_samples)
    mixed_loader = _loader(data_dir, cfg["train_splits"], batch_size, True, args.max_samples)
    val_seen_loader = _loader(data_dir, ["val_composed_seen"], batch_size)
    tuple_loader = tuple_val_loader = None
    if cfg.get("tuple_train_split"):
        tuple_train_ds = TupleRemapDataset(os.path.join(data_dir, f"{cfg['tuple_train_split']}.npz"))
        if args.max_tuples:
            tuple_train_ds = Subset(tuple_train_ds, range(min(args.max_tuples, len(tuple_train_ds))))
        tuple_loader = DataLoader(
            tuple_train_ds,
            batch_size=cfg.get("tuple_batch_size", 32), shuffle=True)
        tuple_val_loader = DataLoader(
            TupleRemapDataset(os.path.join(data_dir, f"{cfg['tuple_val_split']}.npz")),
            batch_size=cfg.get("tuple_batch_size", 32))
    stages = [
        ("single_anchor", args.stage1_epochs if args.stage1_epochs is not None else cfg.get("stage1_epochs", 8)),
        ("tuple_decomposition", (args.stage2_epochs if args.stage2_epochs is not None else cfg.get("stage2_epochs", 12)) if tuple_loader else 0),
        ("direct_finetune", args.stage3_epochs if args.stage3_epochs is not None else cfg.get("stage3_epochs", 10)),
    ]
    run_dir = os.path.join("results", cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))
    metadata = {**common_metadata(cfg, args.config), "config_hash": config_hash(cfg)}
    best_exact, best_loss, best_epoch, epoch = -1.0, float("inf"), None, 0
    with open(os.path.join(run_dir, "train_log.jsonl"), "w") as log:
        for stage, stage_epochs in stages:
            for _ in range(stage_epochs):
                epoch += 1
                started = time.time()
                direct_loader = single_loader if stage == "single_anchor" else mixed_loader
                direct_train = _epoch_direct(model, direct_loader, cfg, device, optimizer)
                tuple_train = None
                if stage != "single_anchor" and tuple_loader is not None:
                    tuple_train = _epoch_tuple(model, tuple_loader, cfg, device, optimizer)
                val = _seen_metrics(model, val_seen_loader, device)
                tuple_val = _epoch_tuple(model, tuple_val_loader, cfg, device) if tuple_val_loader else None
                row = {"epoch": epoch, "stage": stage, "direct_train_loss": direct_train,
                       "tuple_train": tuple_train, "tuple_val": tuple_val,
                       "val_composed_seen_exact": val["exact_match"],
                       "val_composed_seen_macro_f1": val["macro_f1"]}
                log.write(json.dumps(row) + "\n")
                log.flush()
                tie_loss = tuple_val["loss"] if tuple_val else direct_train
                checkpoint = {**metadata, "epoch": epoch, "model_state": model.state_dict(),
                              "model_name": "slot_decomposition",
                              "report_model_name": cfg["report_model_name"], "config": cfg,
                              "seed": seed, "fold_id": cfg["fold_id"],
                              "heldout_composed_pair_id": cfg["heldout_pair_id"],
                              "val_metrics": row}
                torch.save(checkpoint, os.path.join(run_dir, "last.pt"))
                if val["exact_match"] > best_exact or (
                    val["exact_match"] == best_exact and tie_loss < best_loss
                ):
                    best_exact, best_loss, best_epoch = val["exact_match"], tie_loss, epoch
                    torch.save(checkpoint, os.path.join(run_dir, "best_val_composed_seen_exact.pt"))
                print(f"Ep {epoch:3d} {stage:19s} | direct={direct_train:.4f} "
                      f"| tuple={tuple_train['loss'] if tuple_train else float('nan'):.4f} "
                      f"| seen_exact={val['exact_match']:.3f} | {time.time() - started:.1f}s")
    write_json(os.path.join(run_dir, "metrics_val.json"), {
        **metadata, "status": "complete", "best_val_composed_seen_exact": best_exact,
        "best_val_composed_seen_exact_epoch": best_epoch,
        "checkpoint_metric_name": cfg["checkpoint_metric"],
        "checkpoint_metric_value": best_exact,
    })
    write_json(os.path.join(run_dir, "manifest.json"), {
        **metadata, "status": "complete",
        "heldout_pair_names": pair_names(cfg["heldout_pair_id"]) if isinstance(cfg["heldout_pair_id"], int) else None,
        "checkpoint_paths": {"last": "last.pt",
                             "best_val_composed_seen_exact": "best_val_composed_seen_exact.pt"},
    })


if __name__ == "__main__":
    main()
