"""Aggregate final paper experiment outputs into grouped tables."""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


FOLDS = [
    (0, "fold0_goal_topology"),
    (1, "fold1_sensory_action"),
    (2, "fold2_goal_action"),
    (3, "fold3_sensory_topology"),
]
SEEDS = (0, 1, 2)

VARIANT_META = {
    "cnn_diff": {
        "display": "CNN + diff",
        "group": "direct",
        "note": "Local convolutional bias over before/after/diff/absdiff channels.",
    },
    "grid_token_transformer": {
        "display": "Grid-token Transformer",
        "group": "direct",
        "note": "Compact global token mixing over grid-cell features.",
    },
    "cell_graph_gnn": {
        "display": "Cell-graph GNN",
        "group": "direct",
        "note": "4-neighbor message passing over grid cells.",
    },
    "factorized_gate_head": {
        "display": "Factorized gate-head",
        "group": "direct",
        "note": "Naive output-factorized modular gate heads.",
    },
    "primitive_diff_rule": {
        "display": "Primitive diff rule",
        "group": "diagnostic",
        "interpretation": "Task is structurally identifiable from primitive channel differences.",
        "limitation": "Diagnostic rule, not a learned model.",
    },
    "seen_pair_control": {
        "display": "Seen-pair control",
        "group": "diagnostic",
        "interpretation": "Direct labels can fit the heldout pair when the pair is supervised.",
        "limitation": "Privileged learnability control; heldout pair included in training.",
    },
    "oracle_tuple_composer": {
        "display": "Oracle tuple composer",
        "group": "diagnostic",
        "interpretation": "Composition is easy once decomposition is provided.",
        "limitation": "Privileged upper bound, not a deployable model.",
    },
    "evidence_mask_learned_classifier": {
        "display": "Evidence-mask + learned classifier",
        "group": "method",
        "interpretation": "Primitive evidence supervision without faithful evidence-map readout.",
    },
    "evidence_map_noisy_or_unbalanced": {
        "display": "Evidence-map noisy-or, unbalanced",
        "group": "method",
        "interpretation": "Faithful evidence readout without active-cause balancing.",
    },
    "evidence_map_noisy_or_active_balanced": {
        "display": "Evidence-map noisy-or, active-balanced",
        "group": "method",
        "interpretation": "Final method: faithful readout plus active-balanced primitive evidence.",
    },
    "bce_only": {
        "display": "BCE only",
        "group": "ablation",
        "interpretation": "Labels only; no primitive evidence supervision.",
    },
    "evidence_map_noisy_or_active_balanced_no_diff": {
        "display": "Active-balanced noisy-or, no diff",
        "group": "ablation",
        "interpretation": "Optional input ablation removing signed/absolute diff channels.",
    },
}

EXPECTED_VARIANTS = tuple(VARIANT_META)
DIRECT_ROWS = ("cnn_diff", "grid_token_transformer", "cell_graph_gnn", "factorized_gate_head")
DIAGNOSTIC_ROWS = ("primitive_diff_rule", "seen_pair_control", "oracle_tuple_composer")
METHOD_ROWS = (
    "evidence_mask_learned_classifier",
    "evidence_map_noisy_or_unbalanced",
    "evidence_map_noisy_or_active_balanced",
)
ABLATION_ROWS = (
    "bce_only",
    "evidence_mask_learned_classifier",
    "evidence_map_noisy_or_unbalanced",
    "evidence_map_noisy_or_active_balanced",
    "evidence_map_noisy_or_active_balanced_no_diff",
)


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


def _heldout(report):
    return report.get("split_metrics", {}).get("test_composed_heldout", {})


def _active_mask_dice(heldout):
    primitive = heldout.get("primitive_mask", {})
    return _mean([
        row.get("active_only_dice")
        for row in primitive.values()
        if isinstance(row, dict)
    ])


def _row_from_eval(path):
    report = _read(path)
    heldout = _heldout(report)
    per_label = heldout.get("per_label", {})
    topo = per_label.get("map_remap", {})
    sensory = per_label.get("sensory_update", {})
    value = per_label.get("value_remap", {})
    action = per_label.get("action_remap", {})
    fold3 = report.get("fold3_rescue_diagnostics", {})
    faith = heldout.get("faithfulness", {})
    audit = report.get("audit", {})
    return {
        "path": str(path),
        "run_name": report.get("run_name") or path.parent.name,
        "status": report.get("status"),
        "model_key": report.get("model_key"),
        "display_name": report.get("display_name"),
        "fold_id": report.get("fold_id"),
        "fold_name": report.get("fold_name"),
        "seed": report.get("seed"),
        "exact_fixed_0p5": heldout.get("exact_fixed_0p5", heldout.get("exact_match")),
        "exact_val_exact_tuned": heldout.get("exact_val_exact_tuned"),
        "exact_val_f1_tuned": heldout.get("exact_val_f1_tuned"),
        "exact_heldout_exact_tuned_diagnostic": heldout.get("exact_heldout_exact_tuned_diagnostic"),
        "macro_f1": heldout.get("macro_f1"),
        "micro_f1": heldout.get("micro_f1"),
        "per_label": per_label,
        "subset_confusion": heldout.get("subset_confusion", {}),
        "active_mask_dice": _active_mask_dice(heldout),
        "inactive_evidence": heldout.get("inactive_evidence_magnitude"),
        "mask_good_label_good": faith.get("mask_good_label_good"),
        "mask_good_label_bad": faith.get("mask_good_label_bad"),
        "mask_bad_label_good": faith.get("mask_bad_label_good"),
        "mask_bad_label_bad": faith.get("mask_bad_label_bad"),
        "topology_recall": topo.get("recall"),
        "sensory_recall": sensory.get("recall"),
        "value_fp": value.get("fp"),
        "action_fp": action.get("fp"),
        "fold3_topology_max": fold3.get("topology_evidence_max_mean"),
        "fold3_topology_top3": fold3.get("topology_evidence_top3_mean"),
        "fold3_topology_area": fold3.get("topology_evidence_area_gt_0p5_mean"),
        "fold3_topology_active_dice": fold3.get("topology_active_dice"),
        "audit_status": audit.get("status"),
        "audit_passed": bool(audit.get("passed")),
        "audit_errors": audit.get("errors", []),
        "audit_warnings": audit.get("warnings", []),
        "runtime_seconds": report.get("runtime_seconds"),
        "manifest": report.get("manifest", {}),
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


def _expected_keys():
    return {
        (variant, fold_id, seed)
        for variant in EXPECTED_VARIANTS
        for fold_id, _ in FOLDS
        for seed in SEEDS
    }


def _missing_runs(rows):
    have = {(row["model_key"], row["fold_id"], row["seed"]) for row in rows}
    missing = sorted(_expected_keys() - have)
    return [
        {"model_key": model_key, "fold_id": fold_id, "seed": seed}
        for model_key, fold_id, seed in missing
    ]


def _group(rows, model_key):
    return [row for row in rows if row["model_key"] == model_key]


def _fold_values(rows, fold_id, key="exact_fixed_0p5"):
    return [row.get(key) for row in rows if row.get("fold_id") == fold_id]


def _fold_cell(rows, fold_id, key="exact_fixed_0p5"):
    return f"{_fmt(_mean(_fold_values(rows, fold_id, key)))}/{_fmt(_std(_fold_values(rows, fold_id, key)))}"


def _fold_summary(rows, key="exact_fixed_0p5"):
    return {
        f"fold{fold_id}": {
            "mean": _mean(_fold_values(rows, fold_id, key)),
            "std": _std(_fold_values(rows, fold_id, key)),
        }
        for fold_id, _ in FOLDS
    }


def _direct_table(rows):
    table = []
    for key in DIRECT_ROWS:
        group = _group(rows, key)
        meta = VARIANT_META[key]
        table.append({
            "model": meta["display"],
            "fold0_exact_mean_std": _fold_cell(group, 0),
            "fold1_exact_mean_std": _fold_cell(group, 1),
            "fold2_exact_mean_std": _fold_cell(group, 2),
            "fold3_exact_mean_std": _fold_cell(group, 3),
            "overall_exact_mean": _mean([row["exact_fixed_0p5"] for row in group]),
            "overall_exact_std": _std([row["exact_fixed_0p5"] for row in group]),
            "macro_f1_mean": _mean([row["macro_f1"] for row in group]),
            "inductive_bias_note": meta["note"],
        })
    return table


def _diagnostic_table(rows):
    table = []
    for key in DIAGNOSTIC_ROWS:
        group = _group(rows, key)
        meta = VARIANT_META[key]
        table.append({
            "control": meta["display"],
            "mean_exact": _mean([row["exact_fixed_0p5"] for row in group]),
            "fold0_exact": _fmt(_mean(_fold_values(group, 0))),
            "fold1_exact": _fmt(_mean(_fold_values(group, 1))),
            "fold2_exact": _fmt(_mean(_fold_values(group, 2))),
            "fold3_exact": _fmt(_mean(_fold_values(group, 3))),
            "interpretation": meta["interpretation"],
            "limitation": meta["limitation"],
        })
    return table


def _method_table(rows):
    table = []
    for key in METHOD_ROWS:
        group = _group(rows, key)
        meta = VARIANT_META[key]
        table.append({
            "model": meta["display"],
            "fold0_exact_mean_std": _fold_cell(group, 0),
            "fold1_exact_mean_std": _fold_cell(group, 1),
            "fold2_exact_mean_std": _fold_cell(group, 2),
            "fold3_exact_mean_std": _fold_cell(group, 3),
            "overall_exact_mean": _mean([row["exact_fixed_0p5"] for row in group]),
            "overall_exact_std": _std([row["exact_fixed_0p5"] for row in group]),
            "active_only_mask_dice": _mean([row["active_mask_dice"] for row in group]),
            "mask_good_label_bad": _mean([row["mask_good_label_bad"] for row in group]),
            "interpretation": meta["interpretation"],
        })
    return table


def _ablation_table(rows):
    table = []
    for key in ABLATION_ROWS:
        group = _group(rows, key)
        meta = VARIANT_META[key]
        fold3 = [row for row in group if row["fold_id"] == 3]
        table.append({
            "ablation": meta["display"],
            "overall_exact": _mean([row["exact_fixed_0p5"] for row in group]),
            "fold3_exact": _mean([row["exact_fixed_0p5"] for row in fold3]),
            "topology_recall": _mean([row["topology_recall"] for row in fold3]),
            "mask_good_label_bad": _mean([row["mask_good_label_bad"] for row in group]),
            "interpretation": meta["interpretation"],
        })
    return table


def _fold3_table(rows):
    table = []
    for key in ("evidence_map_noisy_or_unbalanced", "evidence_map_noisy_or_active_balanced"):
        group = [row for row in _group(rows, key) if row["fold_id"] == 3]
        confusion = Counter()
        for row in group:
            confusion.update(row.get("subset_confusion") or {})
        table.append({
            "model": VARIANT_META[key]["display"],
            "fold3_exact": _mean([row["exact_fixed_0p5"] for row in group]),
            "topology_recall": _mean([row["topology_recall"] for row in group]),
            "1010_to_1000": confusion.get("1010->1000", 0),
            "1010_to_1010": confusion.get("1010->1010", 0),
            "topology_evidence_max": _mean([row["fold3_topology_max"] for row in group]),
            "topology_top3": _mean([row["fold3_topology_top3"] for row in group]),
            "topology_active_dice": _mean([row["fold3_topology_active_dice"] for row in group]),
            "value_action_fp": sum((row["value_fp"] or 0) + (row["action_fp"] or 0) for row in group),
        })
    return table


def _audit_summary(rows, root):
    warnings = defaultdict(set)
    errors = []
    for row in rows:
        if not row.get("audit_passed"):
            errors.append({
                "run_name": row["run_name"],
                "model_key": row["model_key"],
                "fold_id": row["fold_id"],
                "seed": row["seed"],
                "errors": row.get("audit_errors", []),
            })
        for warning in row.get("audit_warnings", []):
            warnings[warning].add(row["run_name"])
    preflight = {}
    prereq = root / "prereq"
    for path in sorted(prereq.glob("*.json")) if prereq.exists() else []:
        preflight[path.name] = _read(path)
    return {
        "status": "pass" if not errors else "fail",
        "run_audit_errors": errors,
        "unique_run_warnings": [
            {"warning": warning, "n_runs": len(run_names)}
            for warning, run_names in sorted(warnings.items())
        ],
        "preflight": preflight,
    }


def _paper_decision(summary):
    direct_best = max(
        (row["overall_exact_mean"] or 0.0 for row in summary["table_a_direct"]),
        default=0.0,
    )
    final_row = next(row for row in summary["table_c_method"] if "active-balanced" in row["model"])
    final_exact = final_row["overall_exact_mean"] or 0.0
    fold3 = summary["table_e_fold3_rescue"]
    unbalanced = fold3[0]["fold3_exact"] or 0.0
    active = fold3[1]["fold3_exact"] or 0.0
    return {
        "final_method_validated": final_exact > direct_best and active >= 0.70 and active > unbalanced,
        "best_direct_overall_exact": direct_best,
        "final_method_overall_exact": final_exact,
        "fold3_active_balanced_gain": active - unbalanced,
    }


def _figure_data(summary):
    table_a = summary["table_a_direct"]
    table_b = summary["table_b_diagnostic"]
    table_c = summary["table_c_method"]
    table_e = summary["table_e_fold3_rescue"]
    return {
        "figure1_direct_vs_oracle_vs_seen_pair": {
            "direct": table_a,
            "oracle_tuple": [row for row in table_b if row["control"] == "Oracle tuple composer"],
            "seen_pair": [row for row in table_b if row["control"] == "Seen-pair control"],
        },
        "figure2_baselines_vs_final_vs_diagnostics": {
            "direct": table_a,
            "method": table_c,
            "diagnostics": table_b,
        },
        "figure3_fold3_rescue": table_e,
        "figure4_visualization_manifest": "results/paper_final/visualizations/",
    }


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
        "# Paper Final Experiment Tables",
        "",
        "## Table A: Direct learned baselines",
        _markdown_table(summary["table_a_direct"], [
            "model", "fold0_exact_mean_std", "fold1_exact_mean_std",
            "fold2_exact_mean_std", "fold3_exact_mean_std",
            "overall_exact_mean", "overall_exact_std", "macro_f1_mean",
            "inductive_bias_note",
        ]),
        "",
        "## Table B: Diagnostic controls",
        _markdown_table(summary["table_b_diagnostic"], [
            "control", "mean_exact", "fold0_exact", "fold1_exact",
            "fold2_exact", "fold3_exact", "interpretation", "limitation",
        ]),
        "",
        "## Table C: Method comparison",
        _markdown_table(summary["table_c_method"], [
            "model", "fold0_exact_mean_std", "fold1_exact_mean_std",
            "fold2_exact_mean_std", "fold3_exact_mean_std",
            "overall_exact_mean", "overall_exact_std", "active_only_mask_dice",
            "mask_good_label_bad", "interpretation",
        ]),
        "",
        "## Table D: Minimal final-method ablation",
        _markdown_table(summary["table_d_ablation"], [
            "ablation", "overall_exact", "fold3_exact", "topology_recall",
            "mask_good_label_bad", "interpretation",
        ]),
        "",
        "## Table E: Fold3 rescue",
        _markdown_table(summary["table_e_fold3_rescue"], [
            "model", "fold3_exact", "topology_recall", "1010_to_1000",
            "1010_to_1010", "topology_evidence_max", "topology_top3",
            "topology_active_dice", "value_action_fp",
        ]),
    ]
    text = "\n".join(sections) + "\n"
    (root / "paper_final_tables.md").write_text(text)
    (root / "paper_final_tables.txt").write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/paper_final")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    rows = _load_rows(root)
    missing = _missing_runs(rows)
    root.mkdir(parents=True, exist_ok=True)
    (root / "paper_final_missing_runs.json").write_text(json.dumps({
        "status": "complete" if not missing else "missing",
        "expected_runs": len(_expected_keys()),
        "completed_runs": len(rows),
        "missing_runs": missing,
    }, indent=2))
    if missing and not args.allow_incomplete:
        raise SystemExit(f"Refusing to aggregate incomplete final suite: {len(missing)} missing runs")
    audit_summary = _audit_summary(rows, root)
    summary = {
        "status": "complete" if not missing else "incomplete",
        "expected_runs": len(_expected_keys()),
        "completed_runs": len(rows),
        "rows": rows,
        "fold_summary_by_model": {
            key: _fold_summary(_group(rows, key))
            for key in EXPECTED_VARIANTS
        },
        "table_a_direct": _direct_table(rows),
        "table_b_diagnostic": _diagnostic_table(rows),
        "table_c_method": _method_table(rows),
        "table_d_ablation": _ablation_table(rows),
        "table_e_fold3_rescue": _fold3_table(rows),
        "audit_status": audit_summary["status"],
    }
    summary.update(_paper_decision(summary))
    (root / "paper_final_summary.json").write_text(json.dumps(summary, indent=2))
    (root / "paper_final_audit_summary.json").write_text(json.dumps(audit_summary, indent=2))
    (root / "figure_data.json").write_text(json.dumps(_figure_data(summary), indent=2))
    _write_tables(root, summary)
    print(f"Wrote paper-final summary with {len(rows)} rows to {root}")


if __name__ == "__main__":
    main()
