"""Evaluate one direction-finder checkpoint on direct splits and gate contexts."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from models import build_model
from remapbench.data import RemapDataset, TupleRemapDataset
from scripts.evaluate import evaluate_model, get_device
from scripts.evaluate_fepo import evaluate_tuple


SPLITS = ["test_single", "test_composed_seen", "test_composed_heldout",
          "test_larger", "test_noisy"]
GATE_NAMES = ["sensory", "value", "map", "action"]


@torch.no_grad()
def _gate_mean(model, path, device, batch_size, intervention_id=None, pair_id=None):
    raw = np.load(path, allow_pickle=True)
    mask = np.ones(len(raw["intervention_id"]), dtype=bool)
    if intervention_id is not None:
        mask &= raw["intervention_id"] == intervention_id
    if pair_id is not None:
        mask &= raw["intervention_pair_id"] == pair_id
    indices = np.flatnonzero(mask)
    if not len(indices):
        return None
    ds = RemapDataset(path)
    rows = []
    for start in range(0, len(indices), batch_size):
        batch = torch.stack([ds[int(i)]["x"] for i in indices[start:start + batch_size]]).to(device)
        out = model(batch)
        if "gates" not in out:
            return None
        rows.append(out["gates"].cpu().numpy())
    mean = np.concatenate(rows).mean(axis=0)
    return {name: float(mean[i]) for i, name in enumerate(GATE_NAMES)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = get_device(cfg.get("device", "auto"))
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                     weights_only=False)["model_state"])
    report = {"model": cfg["model"], "checkpoint": args.checkpoint, "splits": {}}
    for split in SPLITS:
        path = os.path.join(cfg["data_dir"], f"{split}.npz")
        raw_npz = np.load(path, allow_pickle=True)
        raw = {key: raw_npz[key] for key in raw_npz.files}
        loader = DataLoader(RemapDataset(path), batch_size=cfg.get("batch_size", 128))
        overall, per = evaluate_model(model, loader, device, raw, compute_planning=True)
        report["splits"][split] = {"overall": overall, "per_intervention": per}
    data_dir, batch_size = cfg["data_dir"], cfg.get("batch_size", 128)
    contexts = {
        "goal_only": _gate_mean(model, os.path.join(data_dir, "test_single.npz"),
                                device, batch_size, intervention_id=1),
        "action_only": _gate_mean(model, os.path.join(data_dir, "test_single.npz"),
                                  device, batch_size, intervention_id=3),
        "goal_action_heldout": _gate_mean(
            model, os.path.join(data_dir, "test_composed_heldout.npz"), device, batch_size),
        "sensory_action_seen": _gate_mean(
            model, os.path.join(data_dir, "test_composed_seen.npz"), device, batch_size, pair_id=1),
        "goal_topology_seen": _gate_mean(
            model, os.path.join(data_dir, "test_composed_seen.npz"), device, batch_size, pair_id=0),
    }
    report["gate_contexts"] = contexts
    heldout = contexts["goal_action_heldout"]
    if contexts["goal_only"] and contexts["action_only"] and heldout:
        report["gate_context_drops"] = {
            "value_context_drop": contexts["goal_only"]["value"] - heldout["value"],
            "action_context_drop": contexts["action_only"]["action"] - heldout["action"],
        }
    if cfg["model"].startswith("fepo"):
        report["tuple_diagnostics"] = {}
        for split in ["test_tuple_seen", "test_tuple_heldout"]:
            path = os.path.join(data_dir, f"{split}.npz")
            if os.path.exists(path):
                loader = DataLoader(TupleRemapDataset(path),
                                    batch_size=cfg.get("tuple_batch_size", 32))
                report["tuple_diagnostics"][split] = evaluate_tuple(model, loader, device, 0.5)
    output = args.output or os.path.join(
        "results", os.path.basename(os.path.dirname(args.checkpoint)), "eval_direction.json")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved direction evaluation -> {output}")


if __name__ == "__main__":
    main()
