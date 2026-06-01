"""Uniform direct-validation trainer for the direction-finder pilot."""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from models import build_model
from remapbench.data import RemapDataset
from scripts.train import _compute_loss, eval_epoch, get_device, train_epoch


GATED_MODELS = {
    "gated_erpm", "decoupled_gated_erpm", "factorized_gates",
    "fepo_directonly", "fepo_tuple_supervised",
}


class OrdinaryRows(Dataset):
    """Present tuple rows as ordinary direct-supervision examples."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = dict(self.dataset[idx])
        for key in ("tuple_id", "tuple_role_id", "tuple_pair_id"):
            row.pop(key, None)
        return row


def _datasets(data_dir, names, skip_missing=False):
    parts, used, skipped = [], [], []
    for name in names:
        path = os.path.join(data_dir, f"{name}.npz")
        if not os.path.exists(path):
            if skip_missing:
                skipped.append(name)
                continue
            raise FileNotFoundError(path)
        parts.append(OrdinaryRows(RemapDataset(path)))
        used.append(name)
    if not parts:
        raise ValueError("No datasets selected")
    return (parts[0] if len(parts) == 1 else ConcatDataset(parts)), used, skipped


@torch.no_grad()
def _composed_seen_exact(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        out = model(batch["x"].to(device))
        pred = torch.sigmoid(out["remap_logits"]) > 0.5
        target = batch["target_multihot"].to(device) > 0.5
        correct += int((pred == target).all(dim=1).sum())
        total += len(pred)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_name = cfg["model"]
    device = get_device(cfg.get("device", "auto"))
    model = build_model(model_name).to(device)
    is_gated = model_name in GATED_MODELS
    data_dir = cfg["data_dir"]
    train_names = cfg.get("train_splits", ["train_single"])
    train_names += cfg.get("extra_train_row_splits", [])
    train_ds, used_train, skipped = _datasets(
        data_dir, train_names, cfg.get("skip_missing_extra_train_splits", False))
    val_ds, used_val, _ = _datasets(data_dir, cfg.get("val_splits", ["val_single"]))
    composed_val_ds, _, _ = _datasets(data_dir, ["val_composed_seen"])
    batch_size = cfg.get("batch_size", 128)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    composed_loader = DataLoader(composed_val_ds, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
    train_cfg = {key: cfg.get(key, default) for key, default in {
        "future_weight": 1.0, "value_weight": 1.0, "action_weight": 1.0,
        "type_weight": 1.0, "future_after_weight": 0.0, "value_after_weight": 0.0,
        "gate_sparsity_weight": 0.0, "stability_weight": 0.0,
    }.items()}

    run_dir = os.path.join("results", f"{cfg['run_name']}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))
    print(f"Device: {device} | Model: {model_name} | train={used_train} | skipped={skipped}")
    print(f"Train samples={len(train_ds)} | val samples={len(val_ds)}")

    best_loss, best_exact, best_exact_loss = float("inf"), -1.0, float("inf")
    with open(os.path.join(run_dir, "train_log.jsonl"), "w") as log:
        for epoch in range(1, cfg.get("epochs", 30) + 1):
            started = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, train_cfg, device, is_gated)
            val_loss, f1, comps = eval_epoch(model, val_loader, train_cfg, device, is_gated)
            composed_exact = _composed_seen_exact(model, composed_loader, device)
            ckpt = {"epoch": epoch, "model_state": model.state_dict(), "model_name": model_name,
                    "val_loss": val_loss, "val_composed_seen_exact": composed_exact, "config": cfg}
            torch.save(ckpt, os.path.join(run_dir, "last.pt"))
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(ckpt, os.path.join(run_dir, "best_val_sample_loss.pt"))
            if composed_exact > best_exact or (
                composed_exact == best_exact and val_loss < best_exact_loss
            ):
                best_exact = composed_exact
                best_exact_loss = val_loss
                torch.save(ckpt, os.path.join(run_dir, "best_val_composed_seen_exact.pt"))
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                   "val_composed_seen_exact": composed_exact, **f1, **comps}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"Ep {epoch:3d} | tr={train_loss:.4f} | vl={val_loss:.4f} | "
                  f"seen_exact={composed_exact:.3f} | {time.time() - started:.1f}s")
    with open(os.path.join(run_dir, "metrics_val.json"), "w") as handle:
        json.dump({"model": model_name, "best_val_sample_loss": best_loss,
                   "best_val_composed_seen_exact": best_exact, "used_train_splits": used_train,
                   "skipped_train_splits": skipped}, handle, indent=2)


if __name__ == "__main__":
    main()
