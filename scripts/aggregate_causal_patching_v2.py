"""Aggregate strict causal patching controls across final direct checkpoints."""
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

from scripts.make_final_configs import configs


CONDITIONS = (
    "no_patch", "oracle_gate_only", "correct_update_only_keep_original_gate",
    "correct_gate_plus_update", "same_pathway_unrelated_donor",
    "wrong_pathway_donor", "random_noise_update", "zero_missing_update",
)
MODELS = ("gated", "independent_gate_heads")


def _mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def _std(values):
    values = [value for value in values if value is not None]
    return float(np.std(values)) if values else None


def _reports(root):
    reports, missing = [], []
    for _, cfg in configs():
        if cfg["model"] not in ("gated_erpm", "factorized_gates"):
            continue
        if "seen_upper_bound" in cfg["run_name"]:
            continue
        path = root / cfg["run_name"] / "causal_patching_v2.json"
        if path.exists():
            reports.append(json.loads(path.read_text()))
        else:
            missing.append(str(path))
    return reports, missing


def _decision(flag_table, label_table):
    def rescue(model, condition):
        return next(row["mean_rescue_rate_among_failed"] for row in label_table
                    if row["model"] == model and row["condition"] == condition)
    gate_artifact = all(
        rescue(model, "correct_gate_plus_update") > 0
        and rescue(model, "oracle_gate_only") >= 0.8 * rescue(model, "correct_gate_plus_update")
        and rescue(model, "zero_missing_update") >= 0.8 * rescue(model, "correct_gate_plus_update")
        for model in MODELS)
    if gate_artifact:
        return "PATCHING_GATE_ARTIFACT"
    if any(row["same_pathway_unrelated_matches_correct_rate"] >= 0.5 for row in flag_table):
        return "PATCHING_NONSPECIFIC_PATHWAY_EFFECT"
    if any(row["gate_only_label_artifact_likely_rate"] >= 0.5 for row in flag_table):
        return "PATCHING_GATE_ARTIFACT"
    if all(row["correct_patch_state_specific_rate"] >= 0.5 for row in flag_table):
        return "PATCHING_STRONG_SELECTIVE_CAUSAL"
    return "PATCHING_INCONCLUSIVE"


def _bar(path, labels, series, ylabel):
    x = np.arange(len(labels))
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(11, 5))
    for index, (name, values) in enumerate(series.items()):
        ax.bar(x + index * width, [0 if value is None else value for value in values],
               width, label=name)
    ax.set_xticks(x + width * (len(series) - 1) / 2, labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend()
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
    reports, missing = _reports(root)

    label_table, state_table, flag_table = [], [], []
    for model in MODELS:
        model_rows = [row for row in reports if row["report_model_name"] == model]
        for condition in CONDITIONS:
            metrics = [row["condition_metrics"][condition] for row in model_rows]
            label_table.append({
                "model": model, "condition": condition, "n_runs": len(metrics),
                "mean_exact": _mean([row["exact_match"] for row in metrics]),
                "mean_rescue_rate_among_failed": _mean(
                    [row["rescue_rate_among_failed"] for row in metrics]),
                "std_rescue_rate": _std([row["rescue_rate_among_failed"] for row in metrics]),
            })
            state_table.append({
                "model": model, "condition": condition, "n_runs": len(metrics),
                "delta_future_mse_improvement": _mean(
                    [row["delta_future_mse_improvement"] for row in metrics]),
                "delta_value_mse_improvement": _mean(
                    [row["delta_value_mse_improvement"] for row in metrics]),
                "action_delta_mse_improvement": _mean(
                    [row["action_delta_mse_improvement"] for row in metrics]),
            })
        flags = [row["summary_interpretation_flags"] for row in model_rows]
        flag_table.append({
            "model": model, "n_runs": len(flags),
            "gate_only_label_artifact_likely_rate": _mean(
                [float(row["gate_only_label_artifact_likely"]) for row in flags]),
            "same_pathway_unrelated_matches_correct_rate": _mean(
                [float(row["same_pathway_unrelated_matches_correct"]) for row in flags]),
            "correct_patch_state_specific_rate": _mean(
                [float(row["correct_patch_state_specific"]) for row in flags]),
            "wrong_pathway_low_rate": _mean([float(row["wrong_pathway_low"]) for row in flags]),
            "random_noise_low_rate": _mean([float(row["random_noise_low"]) for row in flags]),
        })
    decision = _decision(flag_table, label_table) if not missing else "PATCHING_INCONCLUSIVE"
    skipped_controls = {
        condition: sum(row["n_skipped_by_condition"][condition] for row in reports)
        for condition in CONDITIONS
    }
    summary = {
        "status": "complete" if not missing else "incomplete",
        "expected_reports": 16, "completed_reports": len(reports), "missing_reports": missing,
        "skipped_controls": skipped_controls,
        "label_rescue": label_table, "state_repair": state_table,
        "interpretation_flags": flag_table, "decision": decision,
    }
    (root / "causal_patching_v2_summary.json").write_text(json.dumps(summary, indent=2))
    with (root / "causal_patching_v2_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(label_table[0]))
        writer.writeheader()
        writer.writerows(label_table)
    lines = ["Strict causal patching v2", "", "Table A - label rescue"]
    lines.extend(json.dumps(row, sort_keys=True) for row in label_table)
    lines += ["", "Table B - state repair"]
    lines.extend(json.dumps(row, sort_keys=True) for row in state_table)
    lines += ["", "Table C - interpretation flags"]
    lines.extend(json.dumps(row, sort_keys=True) for row in flag_table)
    lines += ["", f"Decision: {decision}", f"Missing reports: {len(missing)}"]
    lines.extend(missing)
    (root / "causal_patching_v2_tables.txt").write_text("\n".join(lines) + "\n")

    selected = [row for row in label_table if row["model"] == "gated"]
    _bar(figures / "causal_patching_v2_rescue.png",
         [row["condition"] for row in selected],
         {model: [next(item for item in label_table
                       if item["model"] == model and item["condition"] == row["condition"])
                   ["mean_rescue_rate_among_failed"] for row in selected] for model in MODELS},
         "Mean rescue rate among failed")
    selected = [row for row in state_table if row["model"] == "gated"]
    _bar(figures / "causal_patching_v2_state_mse.png",
         [row["condition"] for row in selected],
         {model: [next(item for item in state_table
                       if item["model"] == model and item["condition"] == row["condition"])
                   ["delta_value_mse_improvement"] for row in selected] for model in MODELS},
         "Mean delta-value MSE improvement")
    print(f"Patching v2 aggregation: {summary['status']} ({len(reports)}/16 reports)")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
