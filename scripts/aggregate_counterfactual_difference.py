"""Aggregate local evidence-mask pilot outputs."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.final_common import write_json


BASELINES = {
    "standard_cnn_direct_mean": 0.1971,
    "shared_gate_direct_mean": 0.1024,
    "independent_gate_heads_direct_mean": 0.0866,
}


def _read(path):
    if not path.exists():
        return None
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text)
    return json.loads(text)


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _std(values):
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return 0.0 if values else None
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def row_for(run_dir):
    report = _read(run_dir / "eval_counterfactual_difference.json")
    audit = _read(run_dir / "audit_counterfactual_difference.json")
    cfg = _read(run_dir / "config.yaml") or {}
    if not report:
        return None
    heldout = report["split_metrics"].get("test_composed_heldout", {})
    seen = report["split_metrics"].get("test_composed_seen", {})
    single = report["split_metrics"].get("test_single", {})
    tuple_metrics = report["split_metrics"].get("test_tuple_heldout_transitions", {})
    legacy_rule_key = "primitive_" + "oracle"
    rule_block = report.get("primitive_diff_rule", report.get(legacy_rule_key, {}))
    rule = rule_block.get("test_composed_heldout", {})
    context = report.get("context_diagnostics", {})
    losses = report.get("loss_weights", {})
    primitive_mask = heldout.get("primitive_mask", {})
    dice_values = [
        row.get("dice") for row in primitive_mask.values()
        if isinstance(row, dict) and row.get("dice") is not None
    ]
    variant = str(report.get("report_model_name") or run_dir.name)
    variant = variant.replace("counterfactual_difference_", "")
    variant = variant.replace("evidence_mask_", "")
    variant = variant.replace("bce_plus_masks", "bce_mask")
    variant = variant.replace("full", "bce_mask_inactive_context")
    confusion = heldout.get("subset_confusion", {})
    wrong = {key: value for key, value in confusion.items() if key.split("->")[0] != key.split("->")[1]}
    faithfulness = heldout.get("faithfulness", {})
    return {
        "run_name": run_dir.name,
        "fold_name": report.get("fold_name"),
        "variant": variant,
        "readout": report.get("readout_type", cfg.get("label_readout")),
        "topk": report.get("readout_topk", cfg.get("readout_topk")),
        "mask_loss": float(losses.get("primitive_mask_weight", cfg.get("primitive_mask_weight", 0.0))) > 0,
        "map_context": float(losses.get("map_context_weight", cfg.get("map_context_weight", 0.0))) > 0,
        "map_union": float(losses.get("map_union_weight", cfg.get("map_union_weight", 0.0))) > 0,
        "detached_map_union": bool(report.get("detach_map_union_target", cfg.get("detach_map_union_target", False))),
        "vector_context": float(losses.get("context_difference_weight", cfg.get("context_difference_weight", 0.0))) > 0,
        "inactive": float(losses.get("inactive_weight", cfg.get("inactive_weight", 0.0))) > 0,
        "conflict_aware": bool(cfg.get("conflict_aware_mask_loss", False)),
        "model": report.get("report_model_name"),
        "seed": report.get("seed"),
        "audit_status": audit.get("status") if audit else None,
        "audit_warnings": len(audit.get("warnings", [])) if audit else None,
        "exact_fixed_0p5": heldout.get("exact_fixed_0p5", heldout.get("exact_match")),
        "exact_val_f1_tuned": heldout.get("exact_val_f1_tuned"),
        "exact_val_exact_tuned": heldout.get("exact_val_exact_tuned", heldout.get("exact_val_tuned")),
        "exact_val_tuned": heldout.get("exact_val_exact_tuned", heldout.get("exact_val_tuned")),
        "heldout_diagnostic_threshold_exact": heldout.get("exact_heldout_exact_tuned_diagnostic", heldout.get("exact_heldout_oracle_threshold_diagnostic")),
        "exact_heldout_exact_tuned_diagnostic": heldout.get("exact_heldout_exact_tuned_diagnostic", heldout.get("exact_heldout_oracle_threshold_diagnostic")),
        "heldout_exact": heldout.get("exact_match"),
        "seen_exact": seen.get("exact_match"),
        "single_exact": single.get("exact_match"),
        "heldout_macro_f1": heldout.get("macro_f1"),
        "tuple_transition_exact": tuple_metrics.get("exact_match"),
        "primitive_diff_rule_exact": rule.get("exact_match"),
        "mask_dice_mean": sum(dice_values) / len(dice_values) if dice_values else None,
        "active_mask_dice": _mean([
            row.get("active_only_dice") for row in primitive_mask.values()
            if isinstance(row, dict) and row.get("active_only_dice") is not None
        ]),
        "inactive_evidence": heldout.get("inactive_evidence_magnitude"),
        "mask_good_label_bad": faithfulness.get("mask_good_label_bad"),
        "mask_good_label_good": faithfulness.get("mask_good_label_good"),
        "mask_bad_label_good": faithfulness.get("mask_bad_label_good"),
        "mask_bad_label_bad": faithfulness.get("mask_bad_label_bad"),
        "most_common_wrong_subset": max(wrong.items(), key=lambda item: item[1])[0] if wrong else None,
        "symmetric_ba_available": context.get("symmetric_ba_available"),
        "context_gap_by_cause": context.get("context_gap_by_cause"),
        "mixed_union_gap_by_cause": context.get("mixed_union_gap_by_cause"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args()
    root = Path(args.results_root)
    run_dirs = sorted(set(root.glob("evidence_mask_*")) | set(root.glob("evidence_readout_*")) | set(root.glob("cdt_*")))
    rows = [row for row in (row_for(path) for path in run_dirs) if row]
    trusted = [
        row for row in rows
        if row["audit_status"] == "pass" and "smoke" not in row["run_name"]
    ]
    unique = {}
    for row in trusted:
        key = (row["fold_name"], row["variant"], row["seed"])
        old = unique.get(key)
        if old is None or row["run_name"].startswith("evidence_mask_"):
            unique[key] = row
    trusted_unique = list(unique.values())
    grouped = {}
    for row in trusted_unique:
        key = row["variant"]
        grouped.setdefault(key, []).append(row["heldout_exact"])
    by_variant = {
        key: {
            "n": len([v for v in values if v is not None]),
            "mean_heldout_exact": (
                sum(v for v in values if v is not None) / len([v for v in values if v is not None])
                if any(v is not None for v in values) else None
            ),
        }
        for key, values in sorted(grouped.items())
    }
    readout_rows = [
        row for row in trusted_unique
        if row["run_name"].startswith("evidence_readout_sieve_")
    ]
    grouped_readout = {}
    for row in readout_rows:
        key = (row["variant"], row["fold_name"], row["readout"])
        grouped_readout.setdefault(key, []).append(row)
    averaged_readout = []
    for (variant, fold_name, readout), group in sorted(grouped_readout.items()):
        averaged_readout.append({
            "variant": variant,
            "fold": fold_name,
            "readout": readout,
            "n": len(group),
            "mean_fixed_exact": _mean([row["exact_fixed_0p5"] for row in group]),
            "std_fixed_exact": _std([row["exact_fixed_0p5"] for row in group]),
            "mean_val_exact_tuned_exact": _mean([row["exact_val_exact_tuned"] for row in group]),
            "std_val_exact_tuned_exact": _std([row["exact_val_exact_tuned"] for row in group]),
            "mean_mask_good_label_bad": _mean([row["mask_good_label_bad"] for row in group]),
            "mean_active_dice": _mean([row["active_mask_dice"] for row in group]),
        })
    disagreement = {}
    for run in run_dirs:
        audit = _read(run / "primitive_diff_rule_audit.json")
        if audit:
            disagreement[run.name] = audit
    summary = {
        "status": "complete" if rows else "empty",
        "baseline_comparison_constants": BASELINES,
        "rows": rows,
        "trusted_rows": trusted,
        "trusted_unique_rows": trusted_unique,
        "by_variant": by_variant,
        "readout_sieve_rows": readout_rows,
        "readout_sieve_averaged": averaged_readout,
        "decision_note": (
            "Evidence-mask supervision is the main method. Context/CDT should stay an ablation "
            "unless it adds robust heldout gains across audited folds."
        ),
    }
    component_decisions = {
        "learned classifier": {
            "decision": "discard",
            "notes": "Fold1 fixed exact stays far below noisy-or; keep only as a baseline.",
        },
        "noisy-or": {
            "decision": "keep",
            "notes": "Best fixed-threshold heldout exact across fold0/fold1 with good seed stability.",
        },
        "topk1": {
            "decision": "discard",
            "notes": "Strong on fold0 but collapses on fold1, especially seed1.",
        },
        "topk3": {
            "decision": "discard",
            "notes": "Weak fixed-threshold exact on both fold0 and fold1; detached union does not rescue it.",
        },
        "topk5": {
            "decision": "discard",
            "notes": "Weakest top-k option; broader pooling hurts fixed exact on the hard fold.",
        },
        "normalized logsumexp": {
            "decision": "discard",
            "notes": "Better than top-k on fold1 but still well below noisy-or at fixed 0.5.",
        },
        "detached map union": {
            "decision": "discard",
            "notes": "Does not improve noisy-or and worsens normalized logsumexp on fold1.",
        },
        "threshold tuning": {
            "decision": "secondary",
            "notes": "Primary metric remains fixed 0.5 exact; val-exact tuning is secondary and F1 tuning is diagnostic.",
        },
    }
    summary["component_decisions"] = component_decisions
    write_json(root / "evidence_mask_sieve_summary.json", summary)
    write_json(root / "evidence_mask_pilot_summary.json", summary)
    write_json(root / "evidence_readout_sieve_summary.json", summary)
    write_json(root / "evidence_mask_disagreement_report.json", {
        "note": "primitive_diff_rule is a rule baseline from primitive channel diffs, not an oracle for target labels.",
        "runs": disagreement,
    })
    csv_path = root / "evidence_mask_sieve_summary.csv"
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "Evidence-mask pilot",
        "",
        "primitive_diff_rule is a rule baseline from primitive channel diffs, not an oracle for target labels.",
        "",
        "Baselines", json.dumps(BASELINES, sort_keys=True), "", "Runs",
    ]
    lines.extend(json.dumps(row, sort_keys=True) for row in rows)
    lines += ["", "Component decisions"]
    lines.extend(f"{key}: {json.dumps(value, sort_keys=True)}" for key, value in component_decisions.items())
    (root / "evidence_mask_sieve_tables.txt").write_text("\n".join(lines) + "\n")
    (root / "evidence_mask_pilot_tables.txt").write_text("\n".join(lines) + "\n")
    readout_lines = [
        "Evidence readout sieve",
        "",
        "Table 1 - main results",
        "fold\tseed\tvariant\treadout\ttopk\tunion\texact_fixed_0p5\texact_val_exact_tuned\texact_val_f1_tuned\texact_heldout_exact_tuned_diagnostic\tseen_exact\tsingle_exact\tactive_mask_dice_mean\tinactive_evidence_mean\tmask_good_label_bad\tmost_common_wrong_subset",
    ]
    for row in readout_rows:
        readout_lines.append("\t".join(str(value) for value in [
            row["fold_name"], row["seed"], row["variant"], row["readout"], row["topk"],
            row["map_union"], row["exact_fixed_0p5"], row["exact_val_exact_tuned"],
            row["exact_val_f1_tuned"], row["exact_heldout_exact_tuned_diagnostic"],
            row["seen_exact"], row["single_exact"], row["active_mask_dice"],
            row["inactive_evidence"], row["mask_good_label_bad"], row["most_common_wrong_subset"],
        ]))
    readout_lines += [
        "",
        "Table 2 - averaged results",
        "variant\tfold\treadout\tmean_fixed_exact\tstd_fixed_exact\tmean_val_exact_tuned_exact\tstd_val_exact_tuned_exact\tmean_mask_good_label_bad\tmean_active_dice",
    ]
    for row in averaged_readout:
        readout_lines.append("\t".join(str(value) for value in [
            row["variant"], row["fold"], row["readout"], row["mean_fixed_exact"],
            row["std_fixed_exact"], row["mean_val_exact_tuned_exact"],
            row["std_val_exact_tuned_exact"], row["mean_mask_good_label_bad"],
            row["mean_active_dice"],
        ]))
    readout_lines += ["", "Table 3 - component decision", "component\tevidence\tdecision\tnotes"]
    for key, value in component_decisions.items():
        readout_lines.append("\t".join([
            key, "see fixed 0.5 exact, seed stability, and mask_good_label_bad",
            value["decision"], value["notes"],
        ]))
    (root / "evidence_readout_sieve_tables.txt").write_text("\n".join(readout_lines) + "\n")
    print(f"Aggregated {len(rows)} evidence-mask runs -> {root / 'evidence_mask_sieve_summary.json'}")


if __name__ == "__main__":
    main()
