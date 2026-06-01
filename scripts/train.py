"""
Training script for ERPM, StandardPredictiveMapCNN, and GatedERPM.
Usage:
    python scripts/train.py --config configs/smoke.yaml
    python scripts/train.py --config configs/smoke.yaml --model erpm --seed 0
"""
import argparse
import json
import os
import sys
import shutil
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from remapbench.data import RemapDataset
from models import build_model


TARGET_NAMES = ["sensory_update", "value_remap", "map_remap", "action_remap"]


def get_device(cfg_device):
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)


def multilabel_f1(preds_bin, targets_bin):
    eps = 1e-8
    tp = (preds_bin & targets_bin).sum(0).float()
    fp = (preds_bin & ~targets_bin).sum(0).float()
    fn = (~preds_bin & targets_bin).sum(0).float()
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1_per = 2 * prec * rec / (prec + rec + eps)
    macro_f1 = float(f1_per.mean())
    micro_tp = tp.sum(); micro_fp = fp.sum(); micro_fn = fn.sum()
    micro_prec = float(micro_tp / (micro_tp + micro_fp + eps))
    micro_rec  = float(micro_tp / (micro_tp + micro_fn + eps))
    micro_f1   = 2 * micro_prec * micro_rec / (micro_prec + micro_rec + eps)
    exact = (preds_bin == targets_bin).all(1).float().mean().item()
    return {"macro_f1": macro_f1, "micro_f1": float(micro_f1), "exact_match": exact,
            "per_label_f1": f1_per.tolist()}


def _compute_loss(out, batch, cfg, device, is_gated):
    df = batch["delta_future"].to(device)
    dv = batch["delta_value"].to(device)
    ad = batch["action_delta"].to(device)
    targets = batch["target_multihot"].to(device)

    loss_df   = F.mse_loss(out["delta_future"], df)
    loss_dv   = F.mse_loss(out["delta_value"],  dv)
    loss_ad   = F.mse_loss(out["action_delta"], ad)
    loss_type = F.binary_cross_entropy_with_logits(out["remap_logits"], targets)

    loss = (cfg["future_weight"] * loss_df +
            cfg["value_weight"]  * loss_dv +
            cfg["action_weight"] * loss_ad +
            cfg["type_weight"]   * loss_type)

    comps = {
        "loss_delta_future": loss_df.item(),
        "loss_delta_value":  loss_dv.item(),
        "loss_action_delta": loss_ad.item(),
        "loss_type":         loss_type.item(),
    }

    if is_gated:
        # future_after / value_after targets
        fa_weight = cfg.get("future_after_weight", 0.0)
        va_weight = cfg.get("value_after_weight",  0.0)
        sp_weight = cfg.get("gate_sparsity_weight", 0.0)
        st_weight = cfg.get("stability_weight",    0.0)

        if fa_weight > 0 and "future_after" in out and "future_after" in batch:
            fa_tgt = batch["future_after"].to(device)
            loss_fa = F.mse_loss(out["future_after"], fa_tgt)
            loss += fa_weight * loss_fa
            comps["loss_future_after"] = loss_fa.item()

        if va_weight > 0 and "value_after" in out and "value_after" in batch:
            va_tgt = batch["value_after"].to(device)
            loss_va = F.mse_loss(out["value_after"], va_tgt)
            loss += va_weight * loss_va
            comps["loss_value_after"] = loss_va.item()

        if sp_weight > 0 and "gates" in out:
            loss_sp = out["gates"].mean()
            loss += sp_weight * loss_sp
            comps["loss_gate_sparsity"] = loss_sp.item()

        if st_weight > 0 and "pathway_updates" in out and "gates" in out:
            # Stability: penalize map pathway update magnitude on sensory-only samples.
            # sensory-only = target [1,0,0,0]: appearance change, no structural remap needed.
            sensory_only = (
                (targets[:, 0] > 0.5) &
                (targets[:, 1] < 0.5) &
                (targets[:, 2] < 0.5) &
                (targets[:, 3] < 0.5)
            )
            if sensory_only.any():
                g_map = out["gates"][:, 2].view(-1, 1, 1, 1)
                u_map = out["pathway_updates"]["map"]
                map_actual = g_map * u_map
                loss_st = (map_actual[sensory_only] ** 2).mean()
            else:
                loss_st = torch.zeros(1, device=device).squeeze()
            loss += st_weight * loss_st
            comps["loss_stability_map_on_sensory"] = loss_st.item()

    return loss, comps


def train_epoch(model, loader, optimizer, cfg, device, is_gated):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        x   = batch["x"].to(device)
        out = model(x)
        loss, _ = _compute_loss(out, batch, cfg, device, is_gated)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(x)
        n += len(x)
    return total / n


@torch.no_grad()
def eval_epoch(model, loader, cfg, device, is_gated):
    model.eval()
    total, n = 0.0, 0
    all_preds, all_targets = [], []
    comp_accum = {}

    for batch in loader:
        x   = batch["x"].to(device)
        out = model(x)
        loss, comps = _compute_loss(out, batch, cfg, device, is_gated)
        total += loss.item() * len(x)
        n += len(x)
        for k, v in comps.items():
            comp_accum[k] = comp_accum.get(k, 0.0) + v * len(x)

        preds_bin   = (torch.sigmoid(out["remap_logits"]) > 0.5)
        targets_bin = batch["target_multihot"].to(device).bool()
        all_preds.append(preds_bin.cpu())
        all_targets.append(targets_bin.cpu())

    preds_all   = torch.cat(all_preds,   0)
    targets_all = torch.cat(all_targets, 0)
    f1_metrics  = multilabel_f1(preds_all, targets_all)
    comp_means  = {k: v / n for k, v in comp_accum.items()}
    return total / n, f1_metrics, comp_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model",  default=None,
                        choices=["erpm", "standard", "standard_cnn", "gated_erpm",
                                 "ungated_latent", "global_plasticity",
                                 "factorized_gates", None])
    parser.add_argument("--seed",   type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = args.model or cfg.get("model", "erpm")
    is_gated   = model_name in (
        "gated_erpm", "ungated_latent", "global_plasticity", "factorized_gates")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(cfg.get("device", "auto"))
    print(f"Device: {device} | Model: {model_name}")

    run_dir = os.path.join("results", cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))

    data_dir = cfg["data_dir"]
    train_splits = cfg.get("train_splits", ["train_single"])
    val_splits   = cfg.get("val_splits", ["val_single"])
    train_parts = [RemapDataset(os.path.join(data_dir, f"{name}.npz"))
                   for name in train_splits]
    val_parts   = [RemapDataset(os.path.join(data_dir, f"{name}.npz"))
                   for name in val_splits]
    train_ds = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    val_ds   = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)
    print(f"Train splits: {train_splits} ({len(train_ds)} samples)")
    print(f"Val splits:   {val_splits} ({len(val_ds)} samples)")

    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 32),
                              shuffle=True,  num_workers=0, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.get("batch_size", 32),
                              shuffle=False, num_workers=0)

    model = build_model(model_name).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name} ({n_params:,} params)")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))

    train_cfg = {
        "future_weight":        cfg.get("future_weight",        1.0),
        "value_weight":         cfg.get("value_weight",         1.0),
        "action_weight":        cfg.get("action_weight",        1.0),
        "type_weight":          cfg.get("type_weight",          1.0),
        "future_after_weight":  cfg.get("future_after_weight",  0.0),
        "value_after_weight":   cfg.get("value_after_weight",   0.0),
        "gate_sparsity_weight": cfg.get("gate_sparsity_weight", 0.0),
        "stability_weight":     cfg.get("stability_weight",     0.0),
    }

    best_val_loss = float("inf")
    log_path = os.path.join(run_dir, "train_log.jsonl")
    log_f = open(log_path, "w")

    epochs = cfg.get("epochs", 10)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, train_cfg, device, is_gated)
        vl_loss, f1, comps = eval_epoch(model, val_loader, train_cfg, device, is_gated)
        elapsed = time.time() - t0

        print(f"Ep {ep:3d}/{epochs} | tr={tr_loss:.4f} | vl={vl_loss:.4f} | "
              f"macro_f1={f1['macro_f1']:.3f} | exact={f1['exact_match']:.3f} | {elapsed:.1f}s")

        row = {"epoch": ep, "train_splits": train_splits, "val_splits": val_splits,
               "n_train_samples": len(train_ds), "n_val_samples": len(val_ds),
               "train_loss": tr_loss, "val_loss": vl_loss, **f1, **comps}
        log_f.write(json.dumps(row) + "\n")
        log_f.flush()

        torch.save({"epoch": ep, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": vl_loss, "f1": f1, "model_name": model_name},
                   os.path.join(run_dir, "last.pt"))
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "val_loss": vl_loss, "f1": f1, "model_name": model_name},
                       os.path.join(run_dir, "best.pt"))

    log_f.close()

    ckpt = torch.load(os.path.join(run_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, best_f1, _ = eval_epoch(model, val_loader, train_cfg, device, is_gated)
    metrics = {"best_val_loss": best_val_loss, "best_epoch": int(ckpt["epoch"]),
               "model": model_name, **best_f1}
    with open(os.path.join(run_dir, "metrics_val.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDone. Best epoch={ckpt['epoch']}, val_loss={best_val_loss:.4f}, "
          f"macro_f1={best_f1['macro_f1']:.3f}")
    print(f"Results saved in {run_dir}/")


if __name__ == "__main__":
    main()
