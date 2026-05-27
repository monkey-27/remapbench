"""
Evaluation script for trained model and heuristic baselines.
Usage:
    python scripts/evaluate.py --config configs/smoke.yaml \
        --checkpoint results/smoke_erpm/best.pt --split test_single
    python scripts/evaluate.py --config configs/smoke.yaml \
        --checkpoint results/smoke_erpm/best.pt --split test_composed
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader

from remapbench.data import RemapDataset
from models.erpm import build_model
from models.baselines import (
    SensoryGatedBaseline, ValueGatedBaseline, MapGatedBaseline,
    ActionGatedBaseline, GlobalPlasticityBaseline, OracleBaseline,
    ALL_BASELINES,
)

TARGET_NAMES = ["sensory_update", "value_remap", "map_remap", "action_remap"]
INTERVENTION_NAMES = {
    0: "sensory_nuisance", 1: "goal_relocation",
    2: "topology_change",  3: "action_change",  4: "composed",
}


def get_device(cfg_device):
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)


def compute_multilabel_metrics(preds, targets):
    """All arrays are float32 numpy [N,4]. Threshold preds at 0.5."""
    preds_b   = (preds   > 0.5).astype(bool)
    targets_b = (targets > 0.5).astype(bool)
    eps = 1e-8
    N = len(preds)

    exact = float((preds_b == targets_b).all(1).mean())

    per_label = {}
    tp_all, fp_all, fn_all = 0, 0, 0
    for i, lname in enumerate(TARGET_NAMES):
        tp = int((preds_b[:, i] & targets_b[:, i]).sum())
        fp = int((preds_b[:, i] & ~targets_b[:, i]).sum())
        fn = int((~preds_b[:, i] & targets_b[:, i]).sum())
        pr = tp / (tp + fp + eps)
        rc = tp / (tp + fn + eps)
        f1 = 2 * pr * rc / (pr + rc + eps)
        per_label[lname] = {"precision": pr, "recall": rc, "f1": float(f1),
                            "tp": tp, "fp": fp, "fn": fn}
        tp_all += tp; fp_all += fp; fn_all += fn

    micro_p = tp_all / (tp_all + fp_all + eps)
    micro_r = tp_all / (tp_all + fn_all + eps)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + eps)
    macro_f1 = float(np.mean([per_label[l]["f1"] for l in TARGET_NAMES]))

    # False structural remap on sensory-only samples:
    #   true = [1,0,0,0] (sensory_update only), pred includes any of value/map/action
    sensory_only = (targets_b[:, 0] & ~targets_b[:, 1] & ~targets_b[:, 2] & ~targets_b[:, 3])
    if sensory_only.sum() > 0:
        false_struct = (preds_b[sensory_only, 1:].any(1)).mean()
    else:
        false_struct = float("nan")

    # Missed remap rate: for each structural label (v/m/a), rate of false negatives
    missed = {}
    for i, lname in enumerate(TARGET_NAMES[1:], 1):
        pos = targets_b[:, i]
        if pos.sum() > 0:
            missed[lname] = float((~preds_b[:, i] & pos).mean())
        else:
            missed[lname] = float("nan")

    return {
        "exact_match": exact,
        "micro_f1": float(micro_f1),
        "macro_f1": macro_f1,
        "per_label": per_label,
        "false_structural_remap_on_sensory": float(false_struct),
        "missed_remap_rates": missed,
        "n_samples": int(N),
    }


def compute_map_mse(pred_maps, true_maps):
    return float(np.mean((pred_maps - true_maps) ** 2))


def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_targets, all_iids = [], [], []
    pred_df, true_df = [], []
    pred_dv, true_dv = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            out = model(x)
            preds = torch.sigmoid(out["remap_logits"]).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(batch["target_multihot"].numpy())
            all_iids.append(batch["intervention_id"].numpy())
            pred_df.append(out["delta_future"].cpu().numpy())
            true_df.append(batch["delta_future"].numpy())
            pred_dv.append(out["delta_value"].cpu().numpy())
            true_dv.append(batch["delta_value"].numpy())

    preds   = np.concatenate(all_preds,   0)
    targets = np.concatenate(all_targets, 0)
    iids    = np.concatenate(all_iids,    0)
    pred_df = np.concatenate(pred_df, 0)
    true_df = np.concatenate(true_df, 0)
    pred_dv = np.concatenate(pred_dv, 0)
    true_dv = np.concatenate(true_dv, 0)

    overall = compute_multilabel_metrics(preds, targets)
    overall["delta_future_mse"] = compute_map_mse(pred_df, true_df)
    overall["delta_value_mse"]  = compute_map_mse(pred_dv, true_dv)

    # Per-intervention breakdown
    per_intervention = {}
    for iid_val in np.unique(iids):
        mask = iids == iid_val
        iname = INTERVENTION_NAMES.get(int(iid_val), f"id{iid_val}")
        per_intervention[iname] = compute_multilabel_metrics(preds[mask], targets[mask])

    return overall, per_intervention


def evaluate_baseline(baseline, scalar_errors, targets, iids):
    if isinstance(baseline, OracleBaseline):
        preds = baseline.predict(scalar_errors, targets=targets)
    else:
        preds = baseline.predict(scalar_errors)

    overall = compute_multilabel_metrics(preds, targets)
    per_intervention = {}
    for iid_val in np.unique(iids):
        mask = iids == iid_val
        iname = INTERVENTION_NAMES.get(int(iid_val), f"id{iid_val}")
        per_intervention[iname] = compute_multilabel_metrics(preds[mask], targets[mask])
    return overall, per_intervention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split",      default="test_single",
                        choices=["test_single", "test_composed", "val_single"])
    parser.add_argument("--model",      default="erpm")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg.get("device", "auto"))
    run_dir = os.path.join("results", cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)

    # Load data
    split_path = os.path.join(cfg["data_dir"], f"{args.split}.npz")
    ds     = RemapDataset(split_path)
    loader = DataLoader(ds, batch_size=cfg.get("batch_size", 64), shuffle=False, num_workers=0)
    print(f"Evaluating on {args.split}: {len(ds)} samples")

    # Collect scalar errors and targets for baseline eval
    raw = np.load(split_path, allow_pickle=True)
    ne  = raw["nuisance_error"]   if "nuisance_error"   in raw else raw.get("sensory_error", np.zeros(len(ds)))
    fve = raw["full_visual_error"] if "full_visual_error" in raw else np.zeros(len(ds))
    scalar_errors = np.stack([ne, fve, raw["future_error"],
                               raw["value_error"], raw["action_error"]], axis=1).astype(np.float32)
    targets = raw["target_multihot"].astype(np.float32)
    iids    = raw["intervention_id"]

    # Neural model evaluation
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(args.model).to(device)
    model.load_state_dict(ckpt["model_state"])
    model_overall, model_per = evaluate_model(model, loader, device)

    print(f"\n=== Neural model ({args.model}) on {args.split} ===")
    print(f"  exact_match={model_overall['exact_match']:.3f} "
          f"micro_f1={model_overall['micro_f1']:.3f} "
          f"macro_f1={model_overall['macro_f1']:.3f}")
    print(f"  false_struct_remap={model_overall['false_structural_remap_on_sensory']:.3f}")
    print(f"  per_label: " +
          "  ".join(f"{l}={v['f1']:.3f}" for l, v in model_overall["per_label"].items()))

    eval_out = {"model": args.model, "split": args.split,
                "overall": model_overall, "per_intervention": model_per}
    out_path = os.path.join(run_dir, f"eval_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(eval_out, f, indent=2)
    print(f"Saved model eval → {out_path}")

    # Heuristic baseline evaluation
    print(f"\n=== Heuristic baselines on {args.split} ===")
    baseline_results = {}
    for BClass in ALL_BASELINES:
        bl = BClass()
        bl_overall, bl_per = evaluate_baseline(bl, scalar_errors, targets, iids)
        baseline_results[bl.name] = {"overall": bl_overall, "per_intervention": bl_per}
        print(f"  {bl.name:25s} exact={bl_overall['exact_match']:.3f} "
              f"micro_f1={bl_overall['micro_f1']:.3f} "
              f"false_struct={bl_overall['false_structural_remap_on_sensory']:.3f}")

    bl_path = os.path.join(run_dir, f"baseline_eval_{args.split}.json")
    with open(bl_path, "w") as f:
        json.dump(baseline_results, f, indent=2)
    print(f"Saved baseline eval → {bl_path}")


if __name__ == "__main__":
    main()
