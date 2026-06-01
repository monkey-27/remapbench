"""Train the privileged CPO tuple diagnostic with explicit tuple-loss provenance."""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from remapbench.data import RemapDataset, TupleRemapDataset
from scripts.final_common import common_metadata, config_hash, load_config, write_json
from scripts.train_cpo import _device, _epoch, _sample_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _device(cfg.get("device", "auto"))
    model = build_model(cfg["model"]).to(device)
    sample_parts = [RemapDataset(os.path.join(cfg["data_dir"], f"{name}.npz"))
                    for name in cfg["train_splits"]]
    val_parts = [RemapDataset(os.path.join(cfg["data_dir"], f"{name}.npz"))
                 for name in cfg["val_splits"]]
    sample_ds = sample_parts[0] if len(sample_parts) == 1 else ConcatDataset(sample_parts)
    val_ds = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)
    tuple_ds = TupleRemapDataset(os.path.join(cfg["data_dir"], f"{cfg['tuple_train_split']}.npz"))
    tuple_val_ds = TupleRemapDataset(os.path.join(cfg["data_dir"], f"{cfg['tuple_val_split']}.npz"))
    sample_loader = DataLoader(sample_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])
    tuple_loader = DataLoader(tuple_ds, batch_size=cfg["tuple_batch_size"], shuffle=True)
    tuple_val_loader = DataLoader(tuple_val_ds, batch_size=cfg["tuple_batch_size"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    run_dir = os.path.join("results", cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))
    metadata = {**common_metadata(cfg, args.config), "config_hash": config_hash(cfg)}
    best, best_epoch = float("inf"), None
    with open(os.path.join(run_dir, "train_log.jsonl"), "w") as log:
        for epoch in range(1, cfg["epochs"] + 1):
            started = time.time()
            sample_train = _sample_epoch(model, sample_loader, cfg, device, optimizer)
            train_metrics = _epoch(model, tuple_loader, cfg, device, optimizer)
            sample_val = _sample_epoch(model, val_loader, cfg, device)
            val_metrics = _epoch(model, tuple_val_loader, cfg, device)
            row = {"epoch": epoch, "sample_train_loss": sample_train,
                   "sample_val_loss": sample_val, "train": train_metrics, "val": val_metrics}
            log.write(json.dumps(row) + "\n")
            log.flush()
            checkpoint = {**metadata, "epoch": epoch, "model_state": model.state_dict(),
                          "model_name": cfg["model"], "report_model_name": cfg["report_model_name"],
                          "config": cfg, "seed": seed, "fold_id": cfg["fold_id"],
                          "heldout_composed_pair_id": cfg["heldout_composed_pair_id"],
                          "val_metrics": val_metrics}
            torch.save(checkpoint, os.path.join(run_dir, "last.pt"))
            if val_metrics["loss"] < best:
                best, best_epoch = val_metrics["loss"], epoch
                torch.save(checkpoint, os.path.join(run_dir, "best_val_tuple_loss.pt"))
            print(f"Ep {epoch:3d} | tuple_val={val_metrics['loss']:.4f} "
                  f"| {time.time() - started:.1f}s")
    write_json(os.path.join(run_dir, "metrics_val.json"), {
        **metadata, "status": "complete", "checkpoint_metric_name": "best_val_tuple_loss",
        "checkpoint_metric_value": best, "best_val_tuple_loss": best,
        "best_val_tuple_loss_epoch": best_epoch,
    })
    write_json(os.path.join(run_dir, "manifest.json"), {
        **metadata, "status": "complete",
        "checkpoint_paths": {"last": "last.pt", "best_val_tuple_loss": "best_val_tuple_loss.pt"},
    })


if __name__ == "__main__":
    main()
