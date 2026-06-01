"""Train CPO on linked base/A/B/AB counterfactual tuples."""
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
from models.cpo import PATHWAYS
from remapbench.data import RemapDataset, TupleRemapDataset
from scripts.train import _compute_loss


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _to_device(role, device):
    return {key: value.to(device) for key, value in role.items() if torch.is_tensor(value)}


def _prediction_loss(out, target, cfg):
    parts = {
        "delta_future": F.mse_loss(out["delta_future"], target["delta_future"]),
        "delta_value": F.mse_loss(out["delta_value"], target["delta_value"]),
        "action_delta": F.mse_loss(out["action_delta"], target["action_delta"]),
        "type": F.binary_cross_entropy_with_logits(out["remap_logits"], target["target_multihot"]),
    }
    if "future_after" in out:
        parts["future_after"] = F.mse_loss(out["future_after"], target["future_after"])
    if "value_after" in out:
        parts["value_after"] = F.mse_loss(out["value_after"], target["value_after"])
    weights = {
        "delta_future": cfg.get("future_weight", 1.0),
        "delta_value": cfg.get("value_weight", 1.0),
        "action_delta": cfg.get("action_weight", 1.0),
        "type": cfg.get("type_weight", 1.0),
        "future_after": cfg.get("future_after_weight", 0.0),
        "value_after": cfg.get("value_after_weight", 0.0),
    }
    return sum(weights[name] * value for name, value in parts.items()), parts


def _reuse_loss(out_a, out_b, out_ab, target_a, target_b):
    terms = []
    for idx, name in enumerate(PATHWAYS):
        mask_a = target_a[:, idx] > 0.5
        mask_b = target_b[:, idx] > 0.5
        if mask_a.any():
            terms.append(F.mse_loss(
                out_ab["pathway_updates"][name][mask_a],
                out_a["pathway_updates"][name][mask_a],
            ))
        if mask_b.any():
            terms.append(F.mse_loss(
                out_ab["pathway_updates"][name][mask_b],
                out_b["pathway_updates"][name][mask_b],
            ))
    if not terms:
        return out_ab["z_updated"].sum() * 0.0
    return torch.stack(terms).mean()


def _loss_and_metrics(model, batch, cfg, device):
    a = _to_device(batch["A"], device)
    b = _to_device(batch["B"], device)
    ab = _to_device(batch["AB"], device)
    out_a, out_b, out_ab = model(a["x"]), model(b["x"]), model(ab["x"])
    out_comp = model.forward_tuple(a["x"], b["x"])
    out_rev = model.forward_tuple(b["x"], a["x"])

    loss_a, _ = _prediction_loss(out_a, a, cfg)
    loss_b, _ = _prediction_loss(out_b, b, cfg)
    loss_ab, _ = _prediction_loss(out_ab, ab, cfg)
    comp_cfg = {**cfg, "type_weight": 0.0}
    loss_comp, _ = _prediction_loss(out_comp, ab, comp_cfg)
    direct = (loss_a + loss_b + loss_ab) / 3.0

    expected_or = torch.maximum(a["target_multihot"], b["target_multihot"])
    loss_gate_or = F.binary_cross_entropy_with_logits(out_comp["remap_logits"], expected_or)
    loss_operator_comp = F.mse_loss(out_comp["z_updated"], out_ab["z_updated"].detach())
    loss_operator_reuse = _reuse_loss(
        out_a, out_b, out_ab, a["target_multihot"], b["target_multihot"])
    loss_comm = F.mse_loss(out_comp["z_updated"], out_rev["z_updated"])
    loss_residual = out_comp.get("tuple_composition_loss", out_comp["z_updated"].sum() * 0.0)

    comp_weight = cfg.get("operator_comp_weight", 1.0)
    total = (
        direct
        + cfg.get("gate_or_weight", 1.0) * loss_gate_or
        + comp_weight * (loss_comp + loss_operator_comp)
        + cfg.get("operator_reuse_weight", 0.5) * loss_operator_reuse
        + cfg.get("commutativity_weight", 0.05) * loss_comm
        + cfg.get("interaction_residual_weight", 0.0) * loss_residual
    )
    with torch.no_grad():
        comp_pred = torch.sigmoid(out_comp["remap_logits"]) > 0.5
        agreement = (comp_pred == (expected_or > 0.5)).all(dim=1).float().mean()
    return total, {
        "loss": float(total.detach()),
        "loss_direct": float(direct.detach()),
        "loss_composed_decode": float(loss_comp.detach()),
        "loss_gate_or": float(loss_gate_or.detach()),
        "loss_operator_comp": float(loss_operator_comp.detach()),
        "loss_operator_reuse": float(loss_operator_reuse.detach()),
        "commutativity_mse": float(loss_comm.detach()),
        "interaction_residual_mse": float(loss_residual.detach()),
        "gate_or_exact_match": float(agreement),
    }


def _epoch(model, loader, cfg, device, optimizer=None):
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
            out = model(x)
            loss, _ = _compute_loss(out, batch, cfg, device, is_gated=True)
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
    if model_name not in ("cpo", "cpo_no_comp"):
        raise ValueError("train_cpo.py supports model cpo or cpo_no_comp")
    model = build_model(model_name).to(device)

    data_dir = cfg["data_dir"]
    train_splits = cfg.get("train_splits", ["train_single"])
    val_splits = cfg.get("val_splits", ["val_single"])
    train_parts = [RemapDataset(os.path.join(data_dir, f"{name}.npz")) for name in train_splits]
    val_parts = [RemapDataset(os.path.join(data_dir, f"{name}.npz")) for name in val_splits]
    sample_train_ds = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    sample_val_ds = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)
    train_ds = TupleRemapDataset(os.path.join(data_dir, f"{cfg['tuple_train_split']}.npz"))
    val_ds = TupleRemapDataset(os.path.join(data_dir, f"{cfg['tuple_val_split']}.npz"))
    sample_train_loader = DataLoader(
        sample_train_ds, batch_size=cfg.get("batch_size", 128), shuffle=True)
    sample_val_loader = DataLoader(
        sample_val_ds, batch_size=cfg.get("batch_size", 128), shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.get("tuple_batch_size", 32), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("tuple_batch_size", 32), shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))

    run_dir = os.path.join("results", cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))
    print(f"Device: {device} | Model: {model_name} | "
          f"train samples={len(sample_train_ds)} | val samples={len(sample_val_ds)} | "
          f"train tuples={len(train_ds)} | val tuples={len(val_ds)}")

    best = float("inf")
    with open(os.path.join(run_dir, "train_log.jsonl"), "w") as log_f:
        for epoch in range(1, cfg.get("epochs", 10) + 1):
            started = time.time()
            sample_train_loss = _sample_epoch(model, sample_train_loader, cfg, device, optimizer)
            train_metrics = _epoch(model, train_loader, cfg, device, optimizer)
            sample_val_loss = _sample_epoch(model, sample_val_loader, cfg, device)
            val_metrics = _epoch(model, val_loader, cfg, device)
            row = {"epoch": epoch, "sample_train_loss": sample_train_loss,
                   "sample_val_loss": sample_val_loss, "train": train_metrics, "val": val_metrics}
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()
            print(
                f"Ep {epoch:3d}/{cfg.get('epochs', 10)} | sample={sample_train_loss:.4f}/{sample_val_loss:.4f} "
                f"| tuple={train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} "
                f"| gate_or={val_metrics['gate_or_exact_match']:.3f} "
                f"| comm={val_metrics['commutativity_mse']:.6f} | {time.time() - started:.1f}s")
            ckpt = {"epoch": epoch, "model_state": model.state_dict(), "model_name": model_name,
                    "config": cfg, "val_metrics": val_metrics}
            torch.save(ckpt, os.path.join(run_dir, "last.pt"))
            if val_metrics["loss"] < best:
                best = val_metrics["loss"]
                torch.save(ckpt, os.path.join(run_dir, "best.pt"))
    with open(os.path.join(run_dir, "metrics_val.json"), "w") as handle:
        json.dump({"best_val_loss": best, "model": model_name}, handle, indent=2)
    print(f"Done. Best val loss={best:.4f}. Results saved in {run_dir}/")


if __name__ == "__main__":
    main()
