"""
Evaluation script for trained model, heuristic baselines, gate ablations, and planning metrics.
Usage:
    python scripts/evaluate.py --config configs/smoke.yaml \
        --checkpoint results/smoke_gated_erpm/best.pt --split test_single
    python scripts/evaluate.py --config configs/smoke.yaml \
        --checkpoint results/smoke_gated_erpm/best.pt --split test_single --gate_ablations
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
from remapbench.planning import compute_planning_metrics
from remapbench.env import directed_path_len, build_transition
from models import build_model
from models.baselines import OracleBaseline, ALL_BASELINES

TARGET_NAMES = ["sensory_update", "value_remap", "map_remap", "action_remap"]
INTERVENTION_NAMES = {
    0: "sensory_nuisance", 1: "goal_relocation",
    2: "topology_change",  3: "action_change",  4: "composed",
}

GATE_ABLATION_MODES = [
    "zero_sensory", "zero_value", "zero_map", "zero_action",
    "all_zero", "all_one", "oracle",
]


def get_device(cfg_device):
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)


def compute_multilabel_metrics(preds, targets, threshold=0.5):
    preds_b   = (preds   > threshold).astype(bool)
    targets_b = (targets > threshold).astype(bool)
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

    sensory_only = (targets_b[:, 0] & ~targets_b[:, 1] & ~targets_b[:, 2] & ~targets_b[:, 3])
    if sensory_only.sum() > 0:
        false_struct = float((preds_b[sensory_only, 1:].any(1)).mean())
    else:
        false_struct = float("nan")

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


def _oracle_path_lens(raw, data_dir):
    """Compute oracle directed path lengths for each sample."""
    from remapbench.env import build_transition, directed_path_len
    grids_after = raw["after_grid"]
    start_xys   = raw["start_xy"]
    goal_xys    = raw["goal_after_xy"]
    W = grids_after.shape[-1]
    lens = []
    for i in range(len(grids_after)):
        T = build_transition(grids_after[i])
        sxy = start_xys[i]
        gxy = goal_xys[i]
        start = (int(sxy[1]), int(sxy[0]))
        goal  = (int(gxy[1]), int(gxy[0]))
        pl = directed_path_len(T, start, goal, W)
        lens.append(pl)
    return lens


def evaluate_model(model, loader, device, raw, threshold=0.5, compute_planning=False,
                   gate_override=None):
    """Evaluate model, optionally with a gate_override for ablation studies.

    When gate_override is set, out["remap_logits"] reflects the overridden gates
    (via _safe_logit), so all reported metrics are consistent with the ablated behavior.
    """
    model.eval()
    all_preds, all_targets, all_iids = [], [], []
    pred_df, true_df = [], []
    pred_dv, true_dv = [], []
    pred_ad, true_ad = [], []
    pred_va_list = []
    gate_list = []
    iid_list_batched = []

    with torch.no_grad():
        for batch in loader:
            x      = batch["x"].to(device)
            tgt_mh = batch["target_multihot"].to(device)

            if gate_override is not None:
                out = model(x, gate_override=gate_override, target_multihot=tgt_mh)
            else:
                out = model(x)

            preds = torch.sigmoid(out["remap_logits"]).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(batch["target_multihot"].numpy())
            iid_b = batch["intervention_id"].numpy()
            all_iids.append(iid_b)
            pred_df.append(out["delta_future"].cpu().numpy())
            true_df.append(batch["delta_future"].numpy())
            pred_dv.append(out["delta_value"].cpu().numpy())
            true_dv.append(batch["delta_value"].numpy())
            pred_ad.append(out["action_delta"].cpu().numpy())
            true_ad.append(batch["action_delta"].numpy())
            if "value_after" in out:
                pred_va_list.append(out["value_after"].cpu().numpy())
            elif "delta_value" in out and "value_before" in batch:
                pred_va_list.append(
                    (batch["value_before"].to(device) + out["delta_value"]).cpu().numpy()
                )
            if "gates" in out:
                gate_list.append(out["gates"].cpu().numpy())
                iid_list_batched.append(iid_b)

    preds      = np.concatenate(all_preds,   0)
    targets    = np.concatenate(all_targets, 0)
    iids       = np.concatenate(all_iids,    0)
    pred_df_np = np.concatenate(pred_df, 0)
    true_df_np = np.concatenate(true_df, 0)
    pred_dv_np = np.concatenate(pred_dv, 0)
    true_dv_np = np.concatenate(true_dv, 0)
    pred_ad_np = np.concatenate(pred_ad, 0)
    true_ad_np = np.concatenate(true_ad, 0)

    overall = compute_multilabel_metrics(preds, targets, threshold)
    overall["delta_future_mse"] = compute_map_mse(pred_df_np, true_df_np)
    overall["delta_value_mse"]  = compute_map_mse(pred_dv_np, true_dv_np)
    overall["action_delta_mse"] = compute_map_mse(pred_ad_np, true_ad_np)

    # Planning metrics using predicted value_after (under the ablated gate if applicable)
    if compute_planning and pred_va_list and raw is not None:
        pred_va_np   = np.concatenate(pred_va_list, 0)  # [N,1,H,W]
        oracle_lens = raw.get("path_len_after")
        if oracle_lens is None:
            oracle_lens = _oracle_path_lens(raw, None)
        value_maps = [pred_va_np[i, 0] for i in range(len(pred_va_np))]
        plan_metrics = compute_planning_metrics(
            list(raw["after_grid"]), list(raw["start_xy"]), list(raw["goal_after_xy"]),
            value_maps, oracle_lens, max_steps=50, penalty=50,
        )
        overall["planning"] = plan_metrics
        oracle_metrics = compute_planning_metrics(
            list(raw["after_grid"]), list(raw["start_xy"]), list(raw["goal_after_xy"]),
            list(raw["value_after"]), oracle_lens, max_steps=50, penalty=50,
        )
        stale_metrics = compute_planning_metrics(
            list(raw["after_grid"]), list(raw["start_xy"]), list(raw["goal_after_xy"]),
            list(raw["value_before"]), oracle_lens, max_steps=50, penalty=50,
        )
        overall["planning_diagnostics"] = {
            "planning_pred_success_rate": plan_metrics["planning_success_rate"],
            "planning_pred_mean_step_regret": plan_metrics["mean_step_regret"],
            "planning_oracle_value_success_rate": oracle_metrics["planning_success_rate"],
            "planning_oracle_value_mean_step_regret": oracle_metrics["mean_step_regret"],
            "planning_stale_value_success_rate": stale_metrics["planning_success_rate"],
            "planning_stale_value_mean_step_regret": stale_metrics["mean_step_regret"],
            "predicted": plan_metrics,
            "oracle_value": oracle_metrics,
            "stale_value": stale_metrics,
        }

    # Mean gate activations per intervention type (reflects final/overridden gates)
    gate_by_itype = {}
    if gate_list:
        all_gates  = np.concatenate(gate_list, 0)
        all_iids_g = np.concatenate(iid_list_batched, 0)
        for iid_val in np.unique(all_iids_g):
            mask  = all_iids_g == iid_val
            iname = INTERVENTION_NAMES.get(int(iid_val), f"id{iid_val}")
            gate_by_itype[iname] = all_gates[mask].mean(0).tolist()
        overall["mean_gates_per_intervention"] = gate_by_itype

    per_intervention = {}
    for iid_val in np.unique(iids):
        mask  = iids == iid_val
        iname = INTERVENTION_NAMES.get(int(iid_val), f"id{iid_val}")
        per_intervention[iname] = compute_multilabel_metrics(
            preds[mask], targets[mask], threshold)

    return overall, per_intervention


def evaluate_baseline(baseline, scalar_errors, targets, iids, threshold=0.5):
    if isinstance(baseline, OracleBaseline):
        preds = baseline.predict(scalar_errors, targets=targets)
    else:
        preds = baseline.predict(scalar_errors)

    overall = compute_multilabel_metrics(preds, targets, threshold)
    per_intervention = {}
    for iid_val in np.unique(iids):
        mask = iids == iid_val
        iname = INTERVENTION_NAMES.get(int(iid_val), f"id{iid_val}")
        per_intervention[iname] = compute_multilabel_metrics(
            preds[mask], targets[mask], threshold)
    return overall, per_intervention


def run_gate_ablations(model, loader, device, raw, threshold=0.5, compute_planning=False):
    """Run each gate ablation mode through evaluate_model for full metrics."""
    results = {}
    for mode in GATE_ABLATION_MODES:
        overall, per_intervention = evaluate_model(
            model, loader, device, raw,
            threshold=threshold,
            compute_planning=compute_planning,
            gate_override=mode,
        )
        results[mode] = {"overall": overall, "per_intervention": per_intervention}

        pm = overall.get("planning", {})
        plan_str = ""
        if pm:
            plan_str = (f" plan_suc={pm.get('planning_success_rate', float('nan')):.3f}"
                        f" regret={pm.get('mean_step_regret', float('nan')):.1f}")
        print(f"  ablation [{mode:15s}]: exact={overall['exact_match']:.3f} "
              f"macro_f1={overall['macro_f1']:.3f} "
              f"false_struct={overall['false_structural_remap_on_sensory']:.3f}"
              f"{plan_str}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        required=True)
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--split",         default="test_single",
                        choices=["test_single", "test_composed", "val_single",
                                 "test_larger", "test_noisy", "train_composed_seen",
                                 "val_composed_seen", "test_composed_seen",
                                 "test_composed_heldout"])
    parser.add_argument("--model",         default=None)
    parser.add_argument("--threshold",     type=float, default=0.5)
    parser.add_argument("--gate_ablations", action="store_true",
                        help="Run gate ablation study (gated_erpm only)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = args.model or cfg.get("model", "erpm")
    is_gated   = (model_name == "gated_erpm")
    has_planning_output = model_name in (
        "erpm", "standard", "standard_cnn",
        "gated_erpm", "ungated_latent", "global_plasticity",
        "factorized_gates", "cpo", "cpo_no_comp",
        "fepo", "fepo_directonly", "fepo_tuple_supervised",
    )

    device  = get_device(cfg.get("device", "auto"))
    run_dir = os.path.join("results", cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)

    split_path = os.path.join(cfg["data_dir"], f"{args.split}.npz")
    if not os.path.exists(split_path):
        print(f"[ERROR] Split not found: {split_path}"); sys.exit(1)

    ds     = RemapDataset(split_path)
    loader = DataLoader(ds, batch_size=cfg.get("batch_size", 64),
                        shuffle=False, num_workers=0)
    print(f"Evaluating on {args.split}: {len(ds)} samples, model={model_name}")

    raw = np.load(split_path, allow_pickle=True)
    raw = {k: raw[k] for k in raw.files}
    ne  = raw.get("nuisance_error", raw.get("sensory_error", np.zeros(len(ds))))
    fve = raw.get("full_visual_error", np.zeros(len(ds)))
    scalar_errors = np.stack([ne, fve, raw["future_error"],
                               raw["value_error"], raw["action_error"]], axis=1).astype(np.float32)
    targets = raw["target_multihot"].astype(np.float32)
    iids    = raw["intervention_id"]

    # Load model
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(model_name).to(device)
    model.load_state_dict(ckpt["model_state"])

    model_overall, model_per = evaluate_model(
        model, loader, device, raw,
        threshold=args.threshold,
        compute_planning=has_planning_output,
    )

    print(f"\n=== Neural model ({model_name}) on {args.split} ===")
    print(f"  exact_match={model_overall['exact_match']:.3f} "
          f"micro_f1={model_overall['micro_f1']:.3f} "
          f"macro_f1={model_overall['macro_f1']:.3f}")
    print(f"  false_struct_remap={model_overall['false_structural_remap_on_sensory']:.3f}")
    print(f"  action_delta_mse={model_overall['action_delta_mse']:.5f}")
    if "planning" in model_overall:
        pm = model_overall["planning"]
        print(f"  planning: success={pm['planning_success_rate']:.3f} "
              f"regret={pm['mean_step_regret']:.2f} failure={pm['failure_rate']:.3f}")
    if "mean_gates_per_intervention" in model_overall:
        print("  mean gates per intervention:")
        for iname, gvals in model_overall["mean_gates_per_intervention"].items():
            print(f"    {iname:25s}: {[f'{v:.3f}' for v in gvals]}")

    eval_out = {"model": model_name, "split": args.split, "threshold": args.threshold,
                "overall": model_overall, "per_intervention": model_per}
    out_path = os.path.join(run_dir, f"eval_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(eval_out, f, indent=2)
    print(f"Saved model eval → {out_path}")

    # Gate ablations
    if args.gate_ablations and is_gated:
        print(f"\n=== Gate ablations on {args.split} ===")
        ablation_results = run_gate_ablations(
            model, loader, device, raw, args.threshold,
            compute_planning=is_gated,
        )
        abl_path = os.path.join(run_dir, f"gate_ablation_eval_{args.split}.json")
        with open(abl_path, "w") as f:
            json.dump(ablation_results, f, indent=2)
        print(f"Saved gate ablation eval → {abl_path}")

    # Heuristic baselines
    print(f"\n=== Heuristic baselines on {args.split} ===")
    baseline_results = {}
    for BClass in ALL_BASELINES:
        bl = BClass()
        bl_overall, bl_per = evaluate_baseline(bl, scalar_errors, targets, iids, args.threshold)
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
