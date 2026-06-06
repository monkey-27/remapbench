"""Aggregate the topology-balance Modal pilot outputs."""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


VARIANTS = [
    ("noisy_or_mask_baseline", "baseline"),
    ("noisy_or_active_balanced_mask", "active-balanced"),
    ("noisy_or_pos_weighted_mask", "pos-weighted"),
    ("noisy_or_active_balanced_pos_weighted_mask", "active-balanced + pos-weighted"),
]


def _read(path):
    with open(path) as handle:
        return json.load(handle)


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _std(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return 0.0 if values else None
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _metric(report, key):
    heldout = report.get("split_metrics", {}).get("test_composed_heldout", {})
    return heldout.get(key)


def _heldout(report):
    return report.get("split_metrics", {}).get("test_composed_heldout", {})


def _audit_status(report):
    audit = report.get("audit", {})
    return {
        "passed": bool(audit.get("passed")),
        "errors": audit.get("errors", []),
        "warnings": audit.get("warnings", []),
    }


def _row_from_eval(path):
    report = _read(path)
    heldout = _heldout(report)
    per_label = heldout.get("per_label", {})
    topology = per_label.get("map_remap", {})
    sensory = per_label.get("sensory_update", {})
    value = per_label.get("value_remap", {})
    action = per_label.get("action_remap", {})
    diag = report.get("fold3_topology_diagnostics", {})
    primitive_mask = heldout.get("primitive_mask", {})
    topo_mask = primitive_mask.get("map", primitive_mask.get("map_remap", {}))
    faith = heldout.get("faithfulness", {})
    return {
        "path": str(path),
        "run_name": report.get("run_name") or path.parent.name,
        "status": report.get("status"),
        "variant": report.get("model_key"),
        "display_name": report.get("display_name"),
        "fold_id": report.get("fold_id"),
        "fold_name": report.get("fold_name"),
        "seed": report.get("seed"),
        "mask_loss_balancing": report.get("mask_loss_balancing"),
        "positive_pixel_weight": report.get("positive_pixel_weight"),
        "inactive_cause_weight": report.get("inactive_cause_weight"),
        "exact_fixed_0p5": heldout.get("exact_fixed_0p5", heldout.get("exact_match")),
        "exact_val_exact_tuned": heldout.get("exact_val_exact_tuned"),
        "exact_heldout_exact_tuned_diagnostic": heldout.get("exact_heldout_exact_tuned_diagnostic"),
        "macro_f1": heldout.get("macro_f1"),
        "micro_f1": heldout.get("micro_f1"),
        "subset_confusion": heldout.get("subset_confusion", {}),
        "topology_recall": topology.get("recall"),
        "topology_precision": topology.get("precision"),
        "topology_f1": topology.get("f1"),
        "sensory_recall": sensory.get("recall"),
        "value_fp": value.get("fp"),
        "action_fp": action.get("fp"),
        "topology_active_dice": topo_mask.get("active_only_dice"),
        "topology_evidence_max_mean": diag.get("topology_evidence_max_mean"),
        "topology_evidence_top3_mean": diag.get("topology_evidence_top3_mean"),
        "topology_evidence_area_gt_0p5_mean": diag.get("topology_evidence_area_gt_0p5_mean"),
        "topology_positive_quantiles": diag.get("topology_positive_quantiles"),
        "inactive_value_evidence_max_mean": diag.get("inactive_value_evidence_max_mean"),
        "inactive_action_evidence_max_mean": diag.get("inactive_action_evidence_max_mean"),
        "mask_good_label_bad": faith.get("mask_good_label_bad"),
        "mask_bad_label_bad": faith.get("mask_bad_label_bad"),
        "audit": _audit_status(report),
        "runtime_seconds": report.get("runtime_seconds"),
    }


def _load_rows(root):
    rows = []
    for path in sorted((root / "runs").glob("*/eval_*.json")):
        if path.name == "eval_counterfactual_difference.json":
            continue
        row = _row_from_eval(path)
        if row.get("status") == "complete":
            rows.append(row)
    return rows


def _by_variant(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    return grouped


def _fold_cell(rows, fold_id, key="exact_fixed_0p5"):
    vals = [row.get(key) for row in rows if row.get("fold_id") == fold_id]
    return f"{_fmt(_mean(vals))}/{_fmt(_std(vals))}"


def _table1(rows):
    table = []
    for key, label in VARIANTS:
        group = [row for row in rows if row["variant"] == key]
        table.append({
            "variant": label,
            "fold0_mean_std": _fold_cell(group, 0),
            "fold1_mean_std": _fold_cell(group, 1),
            "fold2_mean_std": _fold_cell(group, 2),
            "fold3_mean_std": _fold_cell(group, 3),
            "overall_mean": _mean([row["exact_fixed_0p5"] for row in group]),
            "overall_std": _std([row["exact_fixed_0p5"] for row in group]),
            "val_exact_tuned_mean": _mean([row["exact_val_exact_tuned"] for row in group]),
            "val_exact_tuned_std": _std([row["exact_val_exact_tuned"] for row in group]),
            "heldout_opt_diag_mean": _mean([row["exact_heldout_exact_tuned_diagnostic"] for row in group]),
            "heldout_opt_diag_std": _std([row["exact_heldout_exact_tuned_diagnostic"] for row in group]),
        })
    return table


def _table2(rows):
    table = []
    for key, label in VARIANTS:
        group = [row for row in rows if row["variant"] == key and row["fold_id"] == 3]
        confusion = Counter()
        for row in group:
            confusion.update(row.get("subset_confusion") or {})
        table.append({
            "variant": label,
            "fold3_exact_mean": _mean([row["exact_fixed_0p5"] for row in group]),
            "1010_to_1000": confusion.get("1010->1000", 0),
            "1010_to_1010": confusion.get("1010->1010", 0),
            "topology_recall": _mean([row["topology_recall"] for row in group]),
            "sensory_recall": _mean([row["sensory_recall"] for row in group]),
            "topology_active_dice": _mean([row["topology_active_dice"] for row in group]),
            "topology_max": _mean([row["topology_evidence_max_mean"] for row in group]),
            "topology_top3": _mean([row["topology_evidence_top3_mean"] for row in group]),
            "topology_area_gt_0p5": _mean([row["topology_evidence_area_gt_0p5_mean"] for row in group]),
            "value_fp_count": sum(row["value_fp"] or 0 for row in group),
            "action_fp_count": sum(row["action_fp"] or 0 for row in group),
        })
    return table


def _variant_fold_means(rows, variant):
    return {
        fold: _mean([row["exact_fixed_0p5"] for row in rows if row["variant"] == variant and row["fold_id"] == fold])
        for fold in range(4)
    }


def _table3(rows):
    baseline = _variant_fold_means(rows, "noisy_or_mask_baseline")
    base_fold3 = baseline.get(3)
    table = []
    for key, label in VARIANTS:
        means = _variant_fold_means(rows, key)
        fold3_gain = None if base_fold3 is None or means.get(3) is None else means[3] - base_fold3
        max_drop_0_2 = None
        if all(baseline.get(fold) is not None and means.get(fold) is not None for fold in (0, 1, 2)):
            max_drop_0_2 = max(baseline[fold] - means[fold] for fold in (0, 1, 2))
        fold3_rows = [row for row in rows if row["variant"] == key and row["fold_id"] == 3]
        value_fp = sum(row["value_fp"] or 0 for row in fold3_rows)
        action_fp = sum(row["action_fp"] or 0 for row in fold3_rows)
        topo_recall = _mean([row["topology_recall"] for row in fold3_rows])
        topo_evidence = _mean([row["topology_evidence_top3_mean"] for row in fold3_rows])
        if key == "noisy_or_mask_baseline":
            decision = "baseline"
        elif means.get(3) is None:
            decision = "incomplete"
        elif means[3] >= 0.75 and (max_drop_0_2 is None or max_drop_0_2 <= 0.05) and value_fp + action_fp <= 3:
            decision = "strong_keep"
        elif means[3] >= 0.70 and (max_drop_0_2 is None or max_drop_0_2 <= 0.05) and value_fp + action_fp <= 3:
            decision = "keep"
        elif fold3_gain is not None and fold3_gain > 0.03:
            decision = "topology_gain_but_regression_or_fp"
        else:
            decision = "no_topology_rescue"
        table.append({
            "variant": label,
            "fold3_gain_vs_baseline": fold3_gain,
            "max_drop_folds0_2_vs_baseline": max_drop_0_2,
            "fold3_topology_recall": topo_recall,
            "fold3_topology_top3": topo_evidence,
            "fold3_value_action_fp": value_fp + action_fp,
            "decision": decision,
        })
    return table


def _select_decision(summary):
    keepers = [
        row for row in summary["regression_table"]
        if row["decision"] in ("strong_keep", "keep")
    ]
    if keepers:
        keepers.sort(key=lambda row: (row["decision"] == "strong_keep", row["fold3_gain_vs_baseline"] or 0), reverse=True)
        winner = keepers[0]["variant"]
        return {
            "pilot_ready": True,
            "finalist_decision": f"New finalist: evidence_map_noisy_or + {winner} primitive mask loss.",
            "hard_fails": [],
            "next_fix_if_failed": None,
        }
    return {
        "pilot_ready": False,
        "finalist_decision": "No topology-balance variant met the keep rule.",
        "hard_fails": ["No variant reached fold3 exact >= 0.70 while preserving folds0-2 and value/action false positives."],
        "next_fix_if_failed": "Detached map-union readout for topology/mixed evidence only.",
    }


def _audit_summary(rows):
    errors = []
    warnings = []
    for row in rows:
        audit = row.get("audit") or {}
        if not audit.get("passed"):
            errors.append({"run_name": row["run_name"], "errors": audit.get("errors", [])})
        if audit.get("warnings"):
            warnings.append({"run_name": row["run_name"], "warnings": audit.get("warnings", [])})
    return {"errors": errors, "warnings": warnings}


def _markdown_table(rows, columns):
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


def _write_tables(root, summary):
    sections = [
        "# Topology Balance Pilot",
        "",
        "## Table 1: All-fold main results",
        _markdown_table(summary["main_table"], [
            "variant", "fold0_mean_std", "fold1_mean_std", "fold2_mean_std", "fold3_mean_std",
            "overall_mean", "overall_std", "val_exact_tuned_mean", "val_exact_tuned_std",
            "heldout_opt_diag_mean", "heldout_opt_diag_std",
        ]),
        "",
        "## Table 2: Fold3 topology rescue",
        _markdown_table(summary["fold3_table"], [
            "variant", "fold3_exact_mean", "1010_to_1000", "1010_to_1010",
            "topology_recall", "sensory_recall", "topology_active_dice",
            "topology_max", "topology_top3", "topology_area_gt_0p5",
            "value_fp_count", "action_fp_count",
        ]),
        "",
        "## Table 3: Regression check",
        _markdown_table(summary["regression_table"], [
            "variant", "fold3_gain_vs_baseline", "max_drop_folds0_2_vs_baseline",
            "fold3_topology_recall", "fold3_topology_top3", "fold3_value_action_fp", "decision",
        ]),
    ]
    text = "\n".join(sections) + "\n"
    (root / "topology_balance_tables.md").write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/topology_balance_pilot_modal")
    args = parser.parse_args()
    root = Path(args.root)
    rows = _load_rows(root)
    by_variant = _by_variant(rows)
    summary = {
        "status": "complete" if rows else "empty",
        "n_runs": len(rows),
        "expected_runs": 48,
        "rows": rows,
        "by_variant_counts": {key: len(by_variant.get(key, [])) for key, _ in VARIANTS},
        "main_table": _table1(rows),
        "fold3_table": _table2(rows),
        "regression_table": _table3(rows),
        "audit_summary": _audit_summary(rows),
    }
    summary.update(_select_decision(summary))
    fold3_rows = [row for row in rows if row["fold_id"] == 3]
    (root / "topology_balance_summary.json").write_text(json.dumps(summary, indent=2))
    (root / "fold3_topology_diagnostics.json").write_text(json.dumps(fold3_rows, indent=2))
    _write_tables(root, summary)
    print(f"Wrote topology-balance summary with {len(rows)} complete rows to {root}")


if __name__ == "__main__":
    main()
