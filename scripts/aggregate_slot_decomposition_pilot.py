"""Aggregate slot-decomposition pilot reports into compact decision artifacts."""
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

from scripts.make_slot_decomposition_configs import configs


BASELINE_HELDOUT_EXACT = {
    "standard": 0.1971,
    "gated": 0.1024,
    "independent_gate_heads": 0.0866,
}
REPORT_NAMES = ("eval_slot_decomposition.json", "eval_final.json", "metrics_val.json")
SUMMARY_NAME = "slot_decomposition_pilot_summary"
STRICT_SEEN_MIN = 0.80
STRICT_GAIN_MIN = 0.05
STRICT_RETRIEVAL_MIN = 0.75


def _read(path):
    return json.loads(path.read_text()) if path.exists() else None


def _mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def _std(values):
    values = [value for value in values if value is not None]
    return float(np.std(values)) if values else None


def _first(mapping, *keys):
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _split_metrics(report, split):
    if "split_metrics" in report:
        return report["split_metrics"].get(split, {})
    if "splits" in report:
        row = report["splits"].get(split, {})
        return row.get("overall", row)
    return report.get(split, report.get("metrics", report))


def _row(cfg, report):
    split = "test_composed_seen" if cfg["fold_id"] == "seen_upper_bound" else "test_composed_heldout"
    metrics = _split_metrics(report, split)
    diagnostics = report.get("diagnostics", report.get("slot_diagnostics")) or {}
    return {
        "run_name": cfg["run_name"],
        "fold_id": cfg["fold_id"],
        "fold_name": cfg["fold_name"],
        "variant": cfg["variant"],
        "seed": cfg["seed"],
        "reported_split": split,
        "exact_match": _first(metrics, "exact_match", "heldout_exact_match", "best_val_composed_seen_exact"),
        "macro_f1": _first(metrics, "macro_f1", "heldout_macro_f1"),
        "micro_f1": _first(metrics, "micro_f1", "heldout_micro_f1"),
        "retrieval_top1": diagnostics.get("retrieval_top1"),
        "retrieval_margin": diagnostics.get("retrieval_margin"),
        "component_residual_mse": diagnostics.get("component_residual_mse"),
        "correct_donor_similarity": diagnostics.get("correct_donor_similarity"),
        "wrong_donor_similarity": diagnostics.get("wrong_donor_similarity"),
        "unrelated_donor_similarity": diagnostics.get("same_cause_unrelated_donor_similarity"),
        "attention_overlap": metrics.get("attention_overlap"),
        "slot_vector_cosine_overlap": metrics.get("slot_vector_cosine_overlap"),
        "inactive_residual_norm": metrics.get("inactive_residual_norm"),
        "inactive_routing_weight": metrics.get("inactive_routing_weight"),
        "miss_sensory": metrics.get("miss_sensory"),
        "miss_value": metrics.get("miss_value"),
        "miss_topology": metrics.get("miss_topology", metrics.get("miss_map")),
        "miss_action": metrics.get("miss_action"),
        "false_positive_sensory": metrics.get("false_positive_sensory"),
        "false_positive_value": metrics.get("false_positive_value"),
        "false_positive_topology": metrics.get(
            "false_positive_topology", metrics.get("false_positive_map")
        ),
        "false_positive_action": metrics.get("false_positive_action"),
        "delta_future_mse": metrics.get("delta_future_mse"),
        "delta_value_mse": metrics.get("delta_value_mse"),
        "action_delta_mse": metrics.get("action_delta_mse"),
        "future_after_mse": metrics.get("future_after_mse"),
        "value_after_mse": metrics.get("value_after_mse"),
        "global_residual_mse": metrics.get("global_residual_mse"),
    }


def _group(rows):
    grouped = {}
    for row in rows:
        key = (row["fold_name"], row["variant"])
        grouped.setdefault(key, []).append(row)
    return [{
        "fold_name": fold,
        "variant": variant,
        "n_runs": len(group),
        "mean_exact": _mean([row["exact_match"] for row in group]),
        "std_exact": _std([row["exact_match"] for row in group]),
        "mean_macro_f1": _mean([row["macro_f1"] for row in group]),
        "mean_retrieval_top1": _mean([row["retrieval_top1"] for row in group]),
        "mean_retrieval_margin": _mean([row["retrieval_margin"] for row in group]),
        "mean_component_residual_mse": _mean([row["component_residual_mse"] for row in group]),
        "mean_attention_overlap": _mean([row["attention_overlap"] for row in group]),
        "mean_slot_vector_cosine_overlap": _mean([row["slot_vector_cosine_overlap"] for row in group]),
        "mean_inactive_residual_norm": _mean([row["inactive_residual_norm"] for row in group]),
        "mean_inactive_routing_weight": _mean([row["inactive_routing_weight"] for row in group]),
    } for (fold, variant), group in sorted(grouped.items())]


def _lookup(groups, fold_name, variant):
    for row in groups:
        if row["fold_name"] == fold_name and row["variant"] == variant:
            return row["mean_exact"]
    return None


def _decision(missing, groups, seen):
    if missing:
        return "SLOT_DECOMPOSITION_FAILED"
    seen_exact = seen[0]["exact_match"] if seen else None
    if seen_exact is None or seen_exact < STRICT_SEEN_MIN:
        return "SLOT_DECOMPOSITION_FAILED"
    primary = [_lookup(groups, "fold0_goal_topology", "full"),
               _lookup(groups, "fold2_goal_action", "full")]
    ablations = [_lookup(groups, "fold2_goal_action", name)
                 for name in ("no_component", "no_contrastive", "shared_pooled")]
    if any(value is None for value in primary + ablations):
        return "SLOT_DECOMPOSITION_FAILED"
    baseline = max(BASELINE_HELDOUT_EXACT.values())
    if min(primary) < baseline + STRICT_GAIN_MIN:
        return "SLOT_DECOMPOSITION_FAILED"
    if primary[1] < max(ablations) + STRICT_GAIN_MIN:
        return "SLOT_DECOMPOSITION_SUGGESTIVE"
    retrieval = next(row["mean_retrieval_top1"] for row in groups
                     if row["fold_name"] == "fold2_goal_action" and row["variant"] == "full")
    if retrieval is None or retrieval < STRICT_RETRIEVAL_MIN:
        return "SLOT_DECOMPOSITION_LABEL_ONLY"
    return "SLOT_DECOMPOSITION_STRONG"


def _bar(path, labels, values, ylabel, colors=None):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, [0 if value is None else value for value in values], color=colors)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=22)
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
    rows, missing = [], []
    for _, cfg in configs():
        run = root / cfg["run_name"]
        report = None
        for name in REPORT_NAMES:
            report = _read(run / name)
            if report is not None:
                break
        if report is None:
            missing.append(str(run / REPORT_NAMES[0]))
        else:
            rows.append(_row(cfg, report))
    seen = [row for row in rows if row["fold_id"] == "seen_upper_bound"]
    heldout = [row for row in rows if row["fold_id"] != "seen_upper_bound"]
    groups = _group(heldout)
    decision = _decision(missing, groups, seen)
    summary = {
        "status": "complete" if not missing else "incomplete",
        "expected_jobs": 11,
        "completed_jobs": len(rows),
        "missing_reports": missing,
        "baseline_comparison_constants": BASELINE_HELDOUT_EXACT,
        "strict_thresholds": {"seen_pair_min_exact": STRICT_SEEN_MIN,
                              "min_absolute_gain": STRICT_GAIN_MIN,
                              "min_tuple_retrieval_top1": STRICT_RETRIEVAL_MIN},
        "runs": rows,
        "heldout_groups": groups,
        "seen_pair_sanity": seen,
        "decision": decision,
    }
    (root / f"{SUMMARY_NAME}.json").write_text(json.dumps(summary, indent=2))
    csv_rows = rows or [{"run_name": "", "fold_id": "", "fold_name": "", "variant": "",
                         "seed": "", "reported_split": "", "exact_match": "", "macro_f1": "",
                         "micro_f1": "", "retrieval_top1": "", "retrieval_margin": "",
                         "component_residual_mse": "", "attention_overlap": "",
                         "inactive_residual_norm": ""}]
    with (root / f"{SUMMARY_NAME}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        if rows:
            writer.writerows(rows)
    lines = ["Slot decomposition pilot", "", "Baseline comparison constants",
             json.dumps(BASELINE_HELDOUT_EXACT, sort_keys=True), "", "Heldout groups"]
    lines.extend(json.dumps(row, sort_keys=True) for row in groups)
    lines += ["", "Seen-pair sanity"]
    lines.extend(json.dumps(row, sort_keys=True) for row in seen)
    lines += ["", f"Decision: {decision}", f"Missing reports: {len(missing)}"]
    lines.extend(missing)
    (root / "slot_decomposition_pilot_tables.txt").write_text("\n".join(lines) + "\n")

    exact_labels = [f"{row['fold_name'].replace('fold', 'f')}/{row['variant']}" for row in groups]
    exact_values = [row["mean_exact"] for row in groups]
    for name, value in BASELINE_HELDOUT_EXACT.items():
        exact_labels.append(f"baseline/{name}")
        exact_values.append(value)
    _bar(figures / "slot_decomposition_pilot_exact.png", exact_labels, exact_values,
         "Mean heldout exact match")
    diagnostic_groups = [row for row in groups if row["fold_name"] == "fold2_goal_action"]
    _bar(figures / "slot_decomposition_pilot_diagnostics.png",
         [row["variant"] for row in diagnostic_groups],
         [row["mean_retrieval_top1"] for row in diagnostic_groups],
         "Mean tuple slot retrieval top-1")
    print(f"Slot decomposition aggregation: {summary['status']}")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
