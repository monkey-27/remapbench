"""Evaluate one final direct checkpoint with a stable cross-model JSON schema."""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from remapbench.data import RemapDataset
from scripts.evaluate import evaluate_model, get_device
from scripts.final_common import (
    common_metadata, load_config, stable_pathway_metrics, validate_checkpoint, write_json,
)


SPLITS = ["test_single", "test_composed_seen", "test_composed_heldout", "test_larger", "test_noisy"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    validate_checkpoint(cfg, checkpoint)
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    split_metrics = {}
    for split in SPLITS:
        path = os.path.join(cfg["data_dir"], f"{split}.npz")
        with np.load(path, allow_pickle=True) as raw_npz:
            raw = {key: raw_npz[key] for key in raw_npz.files}
        loader = DataLoader(RemapDataset(path), batch_size=cfg.get("batch_size", 128))
        overall, _ = evaluate_model(model, loader, device, raw, compute_planning=True)
        split_metrics[split] = stable_pathway_metrics(overall)
    report = {
        **common_metadata(cfg, args.config), "checkpoint_used": args.checkpoint,
        "split_metrics": split_metrics, "status": "complete",
    }
    output = args.output or os.path.join("results", cfg["run_name"], "eval_final.json")
    write_json(output, report)
    print(f"Saved final evaluation -> {output}")


if __name__ == "__main__":
    main()
