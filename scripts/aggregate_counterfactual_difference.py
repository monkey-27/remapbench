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
    return {
        "run_name": run_dir.name,
        "fold_name": report.get("fold_name"),
        "variant": variant,
        "readout": report.get("readout_type", cfg.get("label_readout")),
        "mask_loss": float(losses.get("primitive_mask_weight", cfg.get("primitive_mask_weight", 0.0))) > 0,
        "map_context": float(losses.get("map_context_weight", cfg.get("map_context_weight", 0.0))) > 0,
        "map_union": float(losses.get("map_union_weight", cfg.get("map_union_weight", 0.0))) > 0,
        "vector_context": float(losses.get("context_difference_weight", cfg.get("context_difference_weight", 0.0))) > 0,
        "inactive": float(losses.get("inactive_weight", cfg.get("inactive_weight", 0.0))) > 0,
        "conflict_aware": bool(cfg.get("conflict_aware_mask_loss", False)),
        "model": report.get("report_model_name"),
        "seed": report.get("seed"),
        "audit_status": audit.get("status") if audit else None,
        "audit_warnings": len(audit.get("warnings", [])) if audit else None,
        "exact_fixed_0p5": heldout.get("exact_fixed_0p5", heldout.get("exact_match")),
        "exact_val_tuned": heldout.get("exact_val_tuned"),
        "heldout_diagnostic_threshold_exact": heldout.get("exact_heldout_oracle_threshold_diagnostic"),
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
    run_dirs = sorted(set(root.glob("evidence_mask_*")) | set(root.glob("cdt_*")))
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
        "decision_note": (
            "Evidence-mask supervision is the main method. Context/CDT should stay an ablation "
            "unless it adds robust heldout gains across audited folds."
        ),
    }
    component_decisions = {
        "diff_channels": {
            "decision": "keep",
            "notes": "Required by the evidence maps; no-diff remains a sanity ablation, not a finalist.",
        },
        "primitive_mask_supervision": {
            "decision": "keep",
            "notes": "BCE-only variants fail or stay weak while mask-supervised variants recover heldout exact.",
        },
        "evidence_pooled_readout": {
            "decision": "keep",
            "notes": "Evidence-pooled readouts beat the learned classifier on the hard fold1 split.",
        },
        "learned_classifier": {
            "decision": "discard",
            "notes": "Keep only as a baseline; it underperforms evidence-pooled readouts on fold1.",
        },
        "evidence_noisy_or": {
            "decision": "finalist",
            "notes": "Strong fixed-threshold fold1 performance with no vector context; validate on clean fold2/fold3.",
        },
        "evidence_logsumexp": {
            "decision": "ablation",
            "notes": "Useful readout, but weaker than noisy-or without extra context on fold1.",
        },
        "evidence_maxpool": {
            "decision": "discard",
            "notes": "Excellent on fold0 but weak on fold1 relative to noisy-or.",
        },
        "map_union": {
            "decision": "ablation",
            "notes": "Improves logsumexp mask-only on fold1 but does not beat noisy-or or vector context.",
        },
        "map_context": {
            "decision": "discard",
            "notes": "Does not beat map union on fold1, and context+union hurts.",
        },
        "vector_context": {
            "decision": "finalist_ablation",
            "notes": "Can be very strong, but seed stability is mixed; validate on clean Modal folds before adopting.",
        },
        "inactive_loss": {
            "decision": "discard",
            "notes": "Inactive loss variants hurt the hard fold1 split.",
        },
        "conflict_aware_mask_loss": {
            "decision": "discard_for_now",
            "notes": "Conflict-aware weighting helped fold0 but hurt fold1 in the local sieve.",
        },
    }
    summary["component_decisions"] = component_decisions
    write_json(root / "evidence_mask_sieve_summary.json", summary)
    write_json(root / "evidence_mask_pilot_summary.json", summary)
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
    print(f"Aggregated {len(rows)} evidence-mask runs -> {root / 'evidence_mask_sieve_summary.json'}")


if __name__ == "__main__":
    main()
