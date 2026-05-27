"""
Training script for ERPM and StandardPredictiveMapCNN.
Usage:
    python scripts/train.py --config configs/smoke.yaml --model erpm --seed 0
    python scripts/train.py --config configs/pilot.yaml --model standard --seed 0
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
from torch.utils.data import DataLoader

from remapbench.data import RemapDataset
from models.erpm import build_model


TARGET_NAMES = ["sensory_update", "value_remap", "map_remap", "action_remap"]


def get_device(cfg_device):
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)


def multilabel_f1(preds_bin, targets_bin):
    """Micro and macro F1 over [N, 4] binary arrays."""
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


def train_epoch(model, loader, optimizer, cfg, device):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        x       = batch["x"].to(device)
        df      = batch["delta_future"].to(device)
        dv      = batch["delta_value"].to(device)
        ad      = batch["action_delta"].to(device)
        targets = batch["target_multihot"].to(device)

        out  = model(x)
        loss = (
            cfg["future_weight"] * F.mse_loss(out["delta_future"], df) +
            cfg["value_weight"]  * F.mse_loss(out["delta_value"],  dv) +
            cfg["action_weight"] * F.mse_loss(out["action_delta"],  ad) +
            cfg["type_weight"]   * F.binary_cross_entropy_with_logits(
                out["remap_logits"], targets)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(x)
        n += len(x)
    return total / n


@torch.no_grad()
def eval_epoch(model, loader, cfg, device):
    model.eval()
    total, n = 0.0, 0
    all_preds, all_targets = [], []
    for batch in loader:
        x       = batch["x"].to(device)
        df      = batch["delta_future"].to(device)
        dv      = batch["delta_value"].to(device)
        ad      = batch["action_delta"].to(device)
        targets = batch["target_multihot"].to(device)

        out  = model(x)
        loss = (
            cfg["future_weight"] * F.mse_loss(out["delta_future"], df) +
            cfg["value_weight"]  * F.mse_loss(out["delta_value"],  dv) +
            cfg["action_weight"] * F.mse_loss(out["action_delta"],  ad) +
            cfg["type_weight"]   * F.binary_cross_entropy_with_logits(
                out["remap_logits"], targets)
        )
        total += loss.item() * len(x)
        n += len(x)
        preds_bin  = (torch.sigmoid(out["remap_logits"]) > 0.5)
        targets_bin = targets.bool()
        all_preds.append(preds_bin.cpu())
        all_targets.append(targets_bin.cpu())

    preds_all   = torch.cat(all_preds,   0)
    targets_all = torch.cat(all_targets, 0)
    f1_metrics  = multilabel_f1(preds_all, targets_all)
    return total / n, f1_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model",  default="erpm", choices=["erpm", "standard"])
    parser.add_argument("--seed",   type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(cfg.get("device", "auto"))
    print(f"Device: {device}")

    run_dir = os.path.join("results", cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))

    data_dir = cfg["data_dir"]
    train_ds = RemapDataset(os.path.join(data_dir, "train_single.npz"))
    val_ds   = RemapDataset(os.path.join(data_dir, "val_single.npz"))
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 32),
                              shuffle=True,  num_workers=0, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.get("batch_size", 32),
                              shuffle=False, num_workers=0)

    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} ({n_params:,} params)")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))

    train_cfg = {
        "future_weight": cfg.get("future_weight", 1.0),
        "value_weight":  cfg.get("value_weight",  1.0),
        "action_weight": cfg.get("action_weight", 1.0),
        "type_weight":   cfg.get("type_weight",   1.0),
    }

    best_val_loss = float("inf")
    log_path = os.path.join(run_dir, "train_log.jsonl")
    log_f = open(log_path, "w")

    epochs = cfg.get("epochs", 10)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, train_cfg, device)
        vl_loss, f1 = eval_epoch(model, val_loader, train_cfg, device)
        elapsed = time.time() - t0

        print(f"Ep {ep:3d}/{epochs} | tr={tr_loss:.4f} | vl={vl_loss:.4f} | "
              f"macro_f1={f1['macro_f1']:.3f} | exact={f1['exact_match']:.3f} | {elapsed:.1f}s")

        row = {"epoch": ep, "train_loss": tr_loss, "val_loss": vl_loss, **f1}
        log_f.write(json.dumps(row) + "\n")
        log_f.flush()

        torch.save({"epoch": ep, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": vl_loss, "f1": f1},
                   os.path.join(run_dir, "last.pt"))
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "val_loss": vl_loss, "f1": f1},
                       os.path.join(run_dir, "best.pt"))

    log_f.close()

    # Final val metrics
    ckpt = torch.load(os.path.join(run_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, best_f1 = eval_epoch(model, val_loader, train_cfg, device)
    metrics = {"best_val_loss": best_val_loss, "best_epoch": int(ckpt["epoch"]), **best_f1}
    with open(os.path.join(run_dir, "metrics_val.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDone. Best epoch={ckpt['epoch']}, val_loss={best_val_loss:.4f}, "
          f"macro_f1={best_f1['macro_f1']:.3f}")
    print(f"Results saved in {run_dir}/")


if __name__ == "__main__":
    main()
