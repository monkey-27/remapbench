"""
Gate sanity check: verifies that gate_override modes actually change gates and outputs.

Usage:
    python scripts/check_gates.py \
        --config configs/smoke.yaml \
        --checkpoint results/smoke_gated_erpm/best.pt \
        --split test_single \
        --model gated_erpm

Asserts:
  - all_zero gates mean < 0.01
  - all_one gates mean > 0.99
  - oracle remap predictions match target_multihot at threshold 0.5
  - at least one override (all_zero/all_one/zero_map) changes delta_future or
    value_after by > 1e-6 vs normal forward pass

Exits with code 1 if any assertion fails.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader

from remapbench.data import RemapDataset
from models import build_model

TARGET_NAMES = ["sensory_update", "value_remap", "map_remap", "action_remap"]


def get_device(cfg_device):
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)


def _mean_abs_diff(a, b):
    return float((a - b).abs().mean())


def main():
    parser = argparse.ArgumentParser(description="Gate override sanity check")
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split",      default="test_single")
    parser.add_argument("--model",      default="gated_erpm")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg.get("device", "auto"))
    split_path = os.path.join(cfg["data_dir"], f"{args.split}.npz")
    if not os.path.exists(split_path):
        print(f"[ERROR] Split not found: {split_path}")
        sys.exit(1)

    ds     = RemapDataset(split_path)
    loader = DataLoader(ds, batch_size=min(32, len(ds)), shuffle=False, num_workers=0)

    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(args.model).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Get first batch only
    batch = next(iter(loader))
    x      = batch["x"].to(device)
    tgt_mh = batch["target_multihot"].to(device)

    modes = ["normal", "all_zero", "all_one", "zero_map", "oracle"]
    outputs = {}

    with torch.no_grad():
        for mode in modes:
            if mode == "normal":
                out = model(x)
            elif mode == "oracle":
                out = model(x, gate_override="oracle", target_multihot=tgt_mh)
            else:
                out = model(x, gate_override=mode)
            outputs[mode] = out

    print("=" * 60)
    print("Gate Sanity Check")
    print("=" * 60)

    # --- Print summary ---
    for mode in modes:
        out = outputs[mode]
        g_mean = float(out["gates"].mean())
        df_diff = _mean_abs_diff(out["delta_future"], outputs["normal"]["delta_future"]) if mode != "normal" else 0.0
        va_diff = _mean_abs_diff(out["value_after"],  outputs["normal"]["value_after"])  if mode != "normal" else 0.0
        print(f"\n[{mode:12s}] mean_gates={g_mean:.4f}  |  "
              f"Δdelta_future={df_diff:.6f}  Δvalue_after={va_diff:.6f}")
        print(f"             gate values (mean per dim): "
              f"{[f'{v:.3f}' for v in out['gates'].mean(0).tolist()]}")

    # --- Oracle: check classification ---
    oracle_preds = (torch.sigmoid(outputs["oracle"]["remap_logits"]) > 0.5).float()
    oracle_match = (oracle_preds == tgt_mh.float()).all(1).float().mean().item()
    print(f"\n[oracle] exact_match vs target_multihot = {oracle_match:.3f}")

    # --- Assertions ---
    failures = []

    all_zero_mean = float(outputs["all_zero"]["gates"].mean())
    if all_zero_mean >= 0.01:
        failures.append(f"FAIL: all_zero gates mean = {all_zero_mean:.4f} (expected < 0.01)")
    else:
        print(f"\n[PASS] all_zero gates mean = {all_zero_mean:.4f} < 0.01")

    all_one_mean = float(outputs["all_one"]["gates"].mean())
    if all_one_mean <= 0.99:
        failures.append(f"FAIL: all_one gates mean = {all_one_mean:.4f} (expected > 0.99)")
    else:
        print(f"[PASS] all_one gates mean = {all_one_mean:.4f} > 0.99")

    if oracle_match < 1.0:
        failures.append(f"FAIL: oracle exact_match = {oracle_match:.3f} (expected 1.0)")
    else:
        print(f"[PASS] oracle exact_match = {oracle_match:.3f}")

    # At least one override should change outputs
    output_changed = False
    for mode in ["all_zero", "all_one", "zero_map"]:
        df_diff = _mean_abs_diff(outputs[mode]["delta_future"], outputs["normal"]["delta_future"])
        va_diff = _mean_abs_diff(outputs[mode]["value_after"],  outputs["normal"]["value_after"])
        if df_diff > 1e-6 or va_diff > 1e-6:
            output_changed = True
            print(f"[PASS] mode={mode} changes outputs "
                  f"(Δdelta_future={df_diff:.2e}, Δvalue_after={va_diff:.2e})")
            break

    if not output_changed:
        failures.append("FAIL: no override (all_zero/all_one/zero_map) changed delta_future or value_after by > 1e-6")

    print("\n" + "=" * 60)
    if failures:
        print("GATE SANITY CHECK FAILED:")
        for msg in failures:
            print(f"  {msg}")
        sys.exit(1)
    else:
        print("ALL GATE SANITY CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
