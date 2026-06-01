"""Train FEPO with ordinary examples plus linked counterfactual tuple losses."""
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
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from models import build_model
from models.fepo import PATHWAYS
from remapbench.data import RemapDataset, TupleRemapDataset
from scripts.train import _compute_loss
from scripts.train_cpo import _prediction_loss, _reuse_loss


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _to_device(role, device):
    return {key: value.to(device) for key, value in role.items() if torch.is_tensor(value)}


def _pooled_evidence_invariance_loss(out_a, out_b, out_ab, target_a, target_b):
    terms = []
    for idx, name in enumerate(PATHWAYS):
        pooled_ab = out_ab["pathway_evidence_features"][name].mean(dim=(2, 3))
        for out_single, target in ((out_a, target_a), (out_b, target_b)):
            active = target[:, idx] > 0.5
            if active.any():
                pooled_single = out_single["pathway_evidence_features"][name].mean(dim=(2, 3))
                terms.append(F.mse_loss(pooled_ab[active], pooled_single[active]))
    if not terms:
        return out_ab["z_updated"].sum() * 0.0
    return torch.stack(terms).mean()


def _inactive_evidence_loss(out_a, out_b, target_a, target_b):
    terms = []
    for idx, name in enumerate(PATHWAYS):
        active_a = target_a[:, idx] > 0.5
        active_b = target_b[:, idx] > 0.5
        if active_a.any():
            terms.append(out_b["evidence_vectors"][name][active_a].square().mean())
        if active_b.any():
            terms.append(out_a["evidence_vectors"][name][active_b].square().mean())
    if not terms:
        return out_a["z_updated"].sum() * 0.0
    return torch.stack(terms).mean()


def _loss_and_metrics(model, batch, cfg, device):
    base = _to_device(batch["base"], device)
    a = _to_device(batch["A"], device)
    b = _to_device(batch["B"], device)
    ab = _to_device(batch["AB"], device)
    out_base, out_a, out_b, out_ab = (
        model(base["x"]), model(a["x"]), model(b["x"]), model(ab["x"]))

    normal_losses = [
        _prediction_loss(out, role, cfg)[0]
        for out, role in ((out_base, base), (out_a, a), (out_b, b), (out_ab, ab))
    ]
    loss_normal = torch.stack(normal_losses).mean()

    loss_pooled_invariance = _pooled_evidence_invariance_loss(
        out_a, out_b, out_ab, a["target_multihot"], b["target_multihot"])

    loss_inactive_evidence = _inactive_evidence_loss(
        out_a, out_b, a["target_multihot"], b["target_multihot"])
    expected_or = torch.maximum(a["target_multihot"], b["target_multihot"])
    loss_gate_or = F.binary_cross_entropy_with_logits(out_ab["remap_logits"], expected_or)
    loss_operator_reuse = _reuse_loss(
        out_a, out_b, out_ab, a["target_multihot"], b["target_multihot"])

    total = (
        loss_normal
        + cfg.get("evidence_invariance_weight", 0.5) * loss_pooled_invariance
        + cfg.get("inactive_evidence_weight", 0.1) * loss_inactive_evidence
        + cfg.get("gate_or_weight", 0.2) * loss_gate_or
        + cfg.get("operator_reuse_weight", 0.1) * loss_operator_reuse
    )
    with torch.no_grad():
        gate_or_exact = (
            (torch.sigmoid(out_ab["remap_logits"]) > 0.5) == (expected_or > 0.5)
        ).all(dim=1).float().mean()
    return total, {
        "loss": float(total.detach()),
        "loss_normal": float(loss_normal.detach()),
        "loss_evidence_invariance": float(loss_pooled_invariance.detach()),
        "loss_inactive_evidence": float(loss_inactive_evidence.detach()),
        "loss_gate_or": float(loss_gate_or.detach()),
        "loss_operator_reuse": float(loss_operator_reuse.detach()),
        "gate_or_exact_match": float(gate_or_exact),
    }


def _tuple_epoch(model, loader, cfg, device, optimizer=None):
    model.train(optimizer is not None)
    totals, n = {}, 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch in loader:
            loss, metrics = _loss_and_metrics(model, batch, cfg, device)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            batch_n = len(batch["tuple_id"])
            n += batch_n
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * batch_n
    return {key: value / n for key, value in totals.items()}


def _sample_epoch(model, loader, cfg, device, optimizer=None):
    model.train(optimizer is not None)
    total, n = 0.0, 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch in loader:
            x = batch["x"].to(device)
            loss, _ = _compute_loss(model(x), batch, cfg, device, is_gated=True)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += float(loss.detach()) * len(x)
            n += len(x)
    return total / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _device(cfg.get("device", "auto"))
    model_name = cfg.get("model", "cpo")
    if model_name not in ("fepo", "fepo_no_evidence_invariance", "fepo_no_operator_reuse"):
        raise ValueError("train_fepo.py supports FEPO architecture variants only")
    model = build_model(model_name).to(device)

    data_dir = cfg["data_dir"]
    train_parts = [
        RemapDataset(os.path.join(data_dir, f"{name}.npz"))
        for name in cfg.get("train_splits", ["train_single"])
    ]
    val_parts = [
        RemapDataset(os.path.join(data_dir, f"{name}.npz"))
        for name in cfg.get("val_splits", ["val_single"])
    ]
    sample_train_ds = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    sample_val_ds = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)
    tuple_train_ds = TupleRemapDataset(
        os.path.join(data_dir, f"{cfg['tuple_train_split']}.npz"))
    tuple_val_ds = TupleRemapDataset(
        os.path.join(data_dir, f"{cfg['tuple_val_split']}.npz"))
    sample_train_loader = DataLoader(
        sample_train_ds, batch_size=cfg.get("batch_size", 128), shuffle=True)
    sample_val_loader = DataLoader(
        sample_val_ds, batch_size=cfg.get("batch_size", 128), shuffle=False)
    tuple_train_loader = DataLoader(
        tuple_train_ds, batch_size=cfg.get("tuple_batch_size", 32), shuffle=True)
    tuple_val_loader = DataLoader(
        tuple_val_ds, batch_size=cfg.get("tuple_batch_size", 32), shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))

    run_dir = os.path.join("results", cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))
    print(f"Device: {device} | Model: {model_name} | "
          f"train samples={len(sample_train_ds)} | val samples={len(sample_val_ds)} | "
          f"train tuples={len(tuple_train_ds)} | val tuples={len(tuple_val_ds)}")

    best = float("inf")
    with open(os.path.join(run_dir, "train_log.jsonl"), "w") as log_f:
        for epoch in range(1, cfg.get("epochs", 10) + 1):
            started = time.time()
            sample_train_loss = _sample_epoch(
                model, sample_train_loader, cfg, device, optimizer)
            train_metrics = _tuple_epoch(model, tuple_train_loader, cfg, device, optimizer)
            sample_val_loss = _sample_epoch(model, sample_val_loader, cfg, device)
            val_metrics = _tuple_epoch(model, tuple_val_loader, cfg, device)
            row = {"epoch": epoch, "sample_train_loss": sample_train_loss,
                   "sample_val_loss": sample_val_loss, "train": train_metrics,
                   "val": val_metrics}
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()
            print(
                f"Ep {epoch:3d}/{cfg.get('epochs', 10)} | "
                f"sample={sample_train_loss:.4f}/{sample_val_loss:.4f} "
                f"| tuple={train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} "
                f"| inv={val_metrics['loss_evidence_invariance']:.6f} "
                f"| suppress={val_metrics['loss_inactive_evidence']:.6f} "
                f"| gate_or={val_metrics['gate_or_exact_match']:.3f} "
                f"| {time.time() - started:.1f}s")
            ckpt = {"epoch": epoch, "model_state": model.state_dict(),
                    "model_name": model_name, "config": cfg, "val_metrics": val_metrics}
            torch.save(ckpt, os.path.join(run_dir, "last.pt"))
            if val_metrics["loss"] < best:
                best = val_metrics["loss"]
                torch.save(ckpt, os.path.join(run_dir, "best.pt"))
    with open(os.path.join(run_dir, "metrics_val.json"), "w") as handle:
        json.dump({"best_val_loss": best, "model": model_name}, handle, indent=2)
    print(f"Done. Best val loss={best:.4f}. Results saved in {run_dir}/")


if __name__ == "__main__":
    main()
