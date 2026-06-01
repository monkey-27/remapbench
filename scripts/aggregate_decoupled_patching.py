"""Aggregate the minimal decoupled-model causal-patching follow-up."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.make_decoupled_patching_configs import configs


CONDITIONS = (
    "no_patch", "gate_only_correct", "update_only_correct",
    "gate_plus_update_correct", "gate_plus_zero_update",
    "same_pathway_unrelated_donor", "wrong_pathway_donor", "random_noise_update",
)


def _read(path):
    return json.loads(path.read_text()) if path.exists() else None


def _mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def _std(values):
    values = [value for value in values if value is not None]
    return float(np.std(values)) if values else None


def _bar(path, labels, values, ylabel):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(labels, [0 if value is None else value for value in values])
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--figures-root", default="figures/final")
    args = parser.parse_args()
    root, figures = Path(args.results_root), Path(args.figures_root)
    root.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    direct, patches, missing = [], [], []
    sanity = None
    for _, cfg in configs():
        run = root / cfg["run_name"]
        evaluation = _read(run / "eval_final.json")
        if evaluation is None:
            missing.append(str(run / "eval_final.json"))
            continue
        if cfg["fold_id"] == "seen_upper_bound":
            sanity = evaluation["split_metrics"]["test_composed_seen"]
            continue
        heldout = evaluation["split_metrics"]["test_composed_heldout"]
        direct.append({
            "fold_id": cfg["fold_id"], "fold_name": cfg["fold_name"], "seed": cfg["seed"],
            "exact_match": heldout["exact_match"], "macro_f1": heldout["macro_f1"],
        })
        patch = _read(run / "causal_patching_v3.json")
        if patch is None:
            missing.append(str(run / "causal_patching_v3.json"))
        else:
            patches.append(patch)

    label_table, state_table = [], []
    for condition in CONDITIONS:
        metrics = [row["condition_metrics"][condition] for row in patches]
        label_table.append({
            "condition": condition, "n_runs": len(metrics),
            "mean_exact": _mean([row["exact_match"] for row in metrics]),
            "mean_rescue_rate_among_failed": _mean(
                [row["rescue_rate_among_failed"] for row in metrics]),
            "std_rescue_rate": _std([row["rescue_rate_among_failed"] for row in metrics]),
        })
        state_table.append({
            "condition": condition, "n_runs": len(metrics),
            "delta_future_mse_improvement": _mean(
                [row["delta_future_mse_improvement"] for row in metrics]),
            "delta_value_mse_improvement": _mean(
                [row["delta_value_mse_improvement"] for row in metrics]),
            "action_delta_mse_improvement": _mean(
                [row["action_delta_mse_improvement"] for row in metrics]),
            "future_after_mse_improvement": _mean(
                [row["future_after_mse_improvement"] for row in metrics]),
            "value_after_mse_improvement": _mean(
                [row["value_after_mse_improvement"] for row in metrics]),
        })
    flag_table = {
        key: _mean([float(row["summary_interpretation_flags"][key]) for row in patches])
        for key in (
            "gate_only_label_artifact_likely", "gate_plus_update_beats_gate_only",
            "update_only_repairs_state", "same_pathway_unrelated_matches_correct",
            "wrong_pathway_low", "noise_low", "selective_state_repair",
        )
    }
    skipped = {
        condition: sum(row["n_skipped_by_condition"][condition] for row in patches)
        for condition in CONDITIONS
    }
    if missing or any(skipped.values()):
        decision = "INCONCLUSIVE"
    elif sanity is None or sanity["exact_match"] < 0.8:
        decision = "DECOUPLED_MODEL_TOO_WEAK"
    elif (flag_table["selective_state_repair"] >= 0.75
          and flag_table["same_pathway_unrelated_matches_correct"] <= 0.25
          and flag_table["gate_only_label_artifact_likely"] <= 0.25):
        decision = "CAUSAL_PATCHING_VALID_MAIN_RESULT"
    else:
        decision = "PATCHING_STILL_GATE_OR_NONSPECIFIC"
    summary = {
        "status": "complete" if not missing else "incomplete",
        "expected_fold_jobs": 8, "completed_fold_jobs": len(direct),
        "expected_seen_pair_sanity_jobs": 1, "completed_seen_pair_sanity_jobs": int(sanity is not None),
        "missing_reports": missing, "skipped_controls": skipped,
        "direct_heldout": direct,
        "prior_direct_model_means": {
            "standard": 0.1971, "gated": 0.1024, "independent_gate_heads": 0.0866},
        "seen_pair_sanity": sanity, "label_rescue": label_table,
        "state_repair": state_table, "interpretation_flags": flag_table, "decision": decision,
    }
    (root / "decoupled_patching_summary.json").write_text(json.dumps(summary, indent=2))
    with (root / "decoupled_patching_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(label_table[0]))
        writer.writeheader()
        writer.writerows(label_table)
    lines = ["Decoupled causal patching", "", "Table 1 - direct heldout"]
    lines.extend(json.dumps(row, sort_keys=True) for row in direct)
    lines += ["", "Table 2 - prior direct model means",
              json.dumps(summary["prior_direct_model_means"], sort_keys=True),
              "", "Table 3 - patching v3 label rescue"]
    lines.extend(json.dumps(row, sort_keys=True) for row in label_table)
    lines += ["", "Table 4 - patching v3 state repair"]
    lines.extend(json.dumps(row, sort_keys=True) for row in state_table)
    lines += ["", "Table 5 - interpretation flags", json.dumps(flag_table, sort_keys=True),
              "", f"Decision: {decision}", f"Missing reports: {len(missing)}"]
    (root / "decoupled_patching_tables.txt").write_text("\n".join(lines) + "\n")
    _bar(figures / "decoupled_patching_rescue.png",
         [row["condition"] for row in label_table],
         [row["mean_rescue_rate_among_failed"] for row in label_table],
         "Mean rescue rate among failed")
    _bar(figures / "decoupled_patching_state_mse.png",
         [row["condition"] for row in state_table],
         [row["delta_value_mse_improvement"] for row in state_table],
         "Mean delta-value MSE improvement")
    print(f"Decoupled aggregation: {summary['status']}")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
