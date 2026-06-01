"""Train one final direct model with fold-aware checkpoint provenance."""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from scripts.final_common import common_metadata, config_hash, load_config, pair_names, write_json
from scripts.train import eval_epoch, get_device, train_epoch
from scripts.train_direction import GATED_MODELS, _composed_seen_exact, _datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_name = cfg["model"]
    device = get_device(cfg.get("device", "auto"))
    model = build_model(model_name).to(device)
    is_gated = model_name in GATED_MODELS
    train_ds, used_train, skipped = _datasets(cfg["data_dir"], cfg["train_splits"])
    val_ds, used_val, _ = _datasets(cfg["data_dir"], cfg["val_splits"])
    composed_ds, _, _ = _datasets(cfg["data_dir"], ["val_composed_seen"])
    batch_size = cfg.get("batch_size", 128)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    composed_loader = DataLoader(composed_ds, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
    loss_cfg = {key: cfg.get(key, default) for key, default in {
        "future_weight": 1.0, "value_weight": 1.0, "action_weight": 1.0,
        "type_weight": 1.0, "future_after_weight": 0.0, "value_after_weight": 0.0,
        "gate_sparsity_weight": 0.0, "stability_weight": 0.0,
    }.items()}
    run_dir = os.path.join("results", cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))
    metadata = common_metadata(cfg, args.config)
    metadata["config_hash"] = config_hash(cfg)
    best_loss, best_exact, best_exact_loss = float("inf"), -1.0, float("inf")
    best_loss_epoch = best_exact_epoch = None
    with open(os.path.join(run_dir, "train_log.jsonl"), "w") as log:
        for epoch in range(1, cfg.get("epochs", 30) + 1):
            started = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, loss_cfg, device, is_gated)
            val_loss, f1, comps = eval_epoch(model, val_loader, loss_cfg, device, is_gated)
            exact = _composed_seen_exact(model, composed_loader, device)
            checkpoint = {
                **metadata, "epoch": epoch, "model_state": model.state_dict(),
                "model_name": model_name, "report_model_name": cfg["report_model_name"],
                "config": cfg, "seed": seed, "fold_id": cfg.get("fold_id"),
                "heldout_composed_pair_id": cfg.get("heldout_composed_pair_id"),
                "val_metrics": {"val_loss": val_loss, "val_composed_seen_exact": exact, **f1},
            }
            torch.save(checkpoint, os.path.join(run_dir, "last.pt"))
            if val_loss < best_loss:
                best_loss, best_loss_epoch = val_loss, epoch
                torch.save(checkpoint, os.path.join(run_dir, "best_val_sample_loss.pt"))
            if exact > best_exact or (exact == best_exact and val_loss < best_exact_loss):
                best_exact, best_exact_loss, best_exact_epoch = exact, val_loss, epoch
                torch.save(checkpoint, os.path.join(run_dir, "best_val_composed_seen_exact.pt"))
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                   "val_composed_seen_exact": exact, **f1, **comps}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"Ep {epoch:3d} | tr={train_loss:.4f} | vl={val_loss:.4f} "
                  f"| seen_exact={exact:.3f} | {time.time() - started:.1f}s")
    checkpoint_paths = {
        "last": "last.pt", "best_val_sample_loss": "best_val_sample_loss.pt",
        "best_val_composed_seen_exact": "best_val_composed_seen_exact.pt",
    }
    metrics = {
        **metadata, "status": "complete", "used_train_splits": used_train,
        "used_val_splits": used_val, "skipped_train_splits": skipped,
        "best_val_sample_loss": best_loss, "best_val_sample_loss_epoch": best_loss_epoch,
        "best_val_composed_seen_exact": best_exact,
        "best_val_composed_seen_exact_epoch": best_exact_epoch,
        "checkpoint_metric_name": cfg["checkpoint_metric"],
        "checkpoint_metric_value": best_exact,
    }
    write_json(os.path.join(run_dir, "metrics_val.json"), metrics)
    write_json(os.path.join(run_dir, "manifest.json"), {
        **metadata, "status": "complete",
        "heldout_pair_names": pair_names(cfg["heldout_composed_pair_id"]),
        "checkpoint_paths": checkpoint_paths,
    })


if __name__ == "__main__":
    main()
