"""Aggregate final Modal evidence-map validation outputs."""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


MAIN_ROWS = [
    ("standard_cnn", "Standard CNN"),
    ("shared_gate", "Shared gate"),
    ("independent_gate_head", "Independent/factorized gate head"),
    ("grid_token_transformer", "Grid-token transformer"),
    ("cell_graph_gnn", "Cell-graph GNN"),
    ("evidence_mask_learned_classifier", "Evidence-mask learned classifier"),
    ("evidence_map_noisy_or", "Evidence-map noisy-or"),
    ("primitive_diff_rule", "Primitive diff rule"),
    ("oracle_tuple_composer", "Oracle tuple composer"),
    ("seen_pair_control", "Seen-pair control"),
]

ABLATION_ROWS = [
    ("bce_only", "BCE only"),
    ("learned_classifier_mask", "learned classifier + mask"),
    ("maxpool_mask", "maxpool + mask"),
    ("logsumexp_norm_mask", "logsumexp_norm + mask"),
    ("topk3_mask", "topk3 + mask"),
    ("noisy_or_mask", "noisy-or + mask"),
    ("noisy_or_mask_no_diff", "noisy-or + mask, no diff channels"),
    ("true_primitive_mask_readout", "true primitive mask readout"),
]

DIAGNOSTIC_ROWS = [
    ("primitive_diff_rule", "Structural primitive-diff solvability diagnostic; not an oracle."),
    ("oracle_tuple_composer", "Privileged decomposed-tuple upper bound."),
    ("seen_pair_control", "Learnability control with heldout pair included in training."),
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
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _most_common(confusions):
    counter = Counter()
    for row in confusions:
        counter.update(row or {})
    wrong = Counter({k: v for k, v in counter.items() if k.split("->")[0] != k.split("->")[1]})
    return wrong.most_common(1)[0][0] if wrong else None


def row_from_eval(path):
    report = _read(path)
    heldout = report.get("split_metrics", {}).get("test_composed_heldout", {})
    primitive_mask = heldout.get("primitive_mask", {})
    faith = heldout.get("faithfulness", {})
    per_label = heldout.get("per_label", {})
    missed = 0
    false_pos = 0
    for item in per_label.values():
        missed += item.get("fn", 0)
        false_pos += item.get("fp", 0)
    return {
        "run_name": report.get("run_name") or path.parent.name,
        "model_key": report.get("model_key"),
        "display_name": report.get("display_name"),
        "fold_name": report.get("fold_name"),
        "fold_id": report.get("fold_id"),
        "seed": report.get("seed"),
        "status": report.get("status"),
        "exact_fixed_0p5": heldout.get("exact_fixed_0p5", heldout.get("exact_match")),
        "exact_val_exact_tuned": heldout.get("exact_val_exact_tuned"),
        "exact_val_f1_tuned": heldout.get("exact_val_f1_tuned"),
        "exact_heldout_exact_tuned_diagnostic": heldout.get("exact_heldout_exact_tuned_diagnostic"),
        "macro_f1": heldout.get("macro_f1"),
        "micro_f1": heldout.get("micro_f1"),
        "active_mask_dice": _mean([
            item.get("active_only_dice")
            for item in primitive_mask.values()
            if isinstance(item, dict)
        ]),
        "inactive_evidence": heldout.get("inactive_evidence_magnitude"),
        "mask_good_label_bad": faith.get("mask_good_label_bad"),
        "mask_good_label_good": faith.get("mask_good_label_good"),
        "subset_confusion": heldout.get("subset_confusion", {}),
        "most_common_wrong_subset": _most_common([heldout.get("subset_confusion", {})]),
        "missed_active_cause_count": missed,
        "false_positive_inactive_cause_count": false_pos,
        "threshold_sensitive_error_count": None
        if heldout.get("exact_val_exact_tuned") is None
        else int(abs(heldout.get("exact_val_exact_tuned") - heldout.get("exact_fixed_0p5", 0.0))
                 * heldout.get("n_samples", 0)),
        "notes": report.get("notes"),
    }


def _group(rows, model_key):
    return [row for row in rows if row["model_key"] == model_key]


def _fold_cell(rows, fold_id):
    vals = [row["exact_fixed_0p5"] for row in rows if row["fold_id"] == fold_id]
    return f"{_fmt(_mean(vals))}/{_fmt(_std(vals))}"


def _main_table(rows):
    table = []
    for key, label in MAIN_ROWS:
        group = _group(rows, key)
        table.append({
            "model_key": key,
            "model": label,
            "fold0_exact_mean_std": _fold_cell(group, 0),
            "fold1_exact_mean_std": _fold_cell(group, 1),
            "fold2_exact_mean_std": _fold_cell(group, 2),
            "fold3_exact_mean_std": _fold_cell(group, 3),
            "overall_mean": _mean([row["exact_fixed_0p5"] for row in group]),
            "overall_std": _std([row["exact_fixed_0p5"] for row in group]),
            "macro_f1_mean": _mean([row["macro_f1"] for row in group]),
            "notes": group[0]["notes"] if group else "",
        })
    return table


def _ablation_table(rows):
    table = []
    for key, label in ABLATION_ROWS:
        group = _group(rows, key)
        table.append({
            "model_key": key,
            "model": label,
            "heldout_exact_mean": _mean([row["exact_fixed_0p5"] for row in group]),
            "heldout_exact_std": _std([row["exact_fixed_0p5"] for row in group]),
            "active_only_mask_dice": _mean([row["active_mask_dice"] for row in group]),
            "inactive_evidence": _mean([row["inactive_evidence"] for row in group]),
            "mask_good_label_bad": _mean([row["mask_good_label_bad"] for row in group]),
            "subset_confusion_summary": _most_common([row["subset_confusion"] for row in group]),
        })
    return table


def _diagnostic_table(rows):
    table = []
    for key, interpretation in DIAGNOSTIC_ROWS:
        group = _group(rows, key)
        table.append({
            "model_key": key,
            "exact": _mean([row["exact_fixed_0p5"] for row in group]),
            "interpretation": interpretation,
            "limitations": group[0]["notes"] if group else "",
        })
    return table


def _error_table(rows):
    table = []
    for row in rows:
        table.append({
            "fold": row["fold_name"],
            "model_key": row["model_key"],
            "most_common_wrong_subset": row["most_common_wrong_subset"],
            "missed_active_cause_count": row["missed_active_cause_count"],
            "false_positive_inactive_cause_count": row["false_positive_inactive_cause_count"],
            "threshold_sensitive_error_count": row["threshold_sensitive_error_count"],
            "mask_good_label_bad": row["mask_good_label_bad"],
        })
    return table


def _markdown_table(rows, columns):
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


def write_tables(root, summary):
    sections = [
        "# Final Evidence-Map Modal Validation",
        "",
        "## Table 1: Direct-vs-oracle comparison",
        _markdown_table(summary["main_table"], [
            "model", "fold0_exact_mean_std", "fold1_exact_mean_std",
            "fold2_exact_mean_std", "fold3_exact_mean_std",
            "overall_mean", "overall_std", "macro_f1_mean", "notes",
        ]),
        "",
        "## Table 2: Final method ablations",
        _markdown_table(summary["ablation_table"], [
            "model", "heldout_exact_mean", "heldout_exact_std",
            "active_only_mask_dice", "inactive_evidence",
            "mask_good_label_bad", "subset_confusion_summary",
        ]),
        "",
        "## Table 3: Diagnostics",
        _markdown_table(summary["diagnostic_table"], [
            "model_key", "exact", "interpretation", "limitations",
        ]),
        "",
        "## Table 4: Error analysis",
        _markdown_table(summary["error_table"], [
            "fold", "model_key", "most_common_wrong_subset",
            "missed_active_cause_count", "false_positive_inactive_cause_count",
            "threshold_sensitive_error_count", "mask_good_label_bad",
        ]),
    ]
    text = "\n".join(sections) + "\n"
    (root / "final_tables.md").write_text(text)
    (root / "final_tables.txt").write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/final_evidence_map_modal")
    args = parser.parse_args()
    root = Path(args.root)
    rows = [row_from_eval(path) for path in sorted((root / "runs").glob("*/eval_*.json"))]
    rows = [row for row in rows if row.get("status") == "complete"]
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model_key"]].append(row)
    summary = {
        "status": "complete" if rows else "empty",
        "n_runs": len(rows),
        "rows": rows,
        "main_table": _main_table(rows),
        "ablation_table": _ablation_table(rows),
        "diagnostic_table": _diagnostic_table(rows),
        "error_table": _error_table(rows),
        "by_model_counts": {key: len(value) for key, value in sorted(by_model.items())},
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "final_summary.json").write_text(json.dumps(summary, indent=2))
    write_tables(root, summary)
    print(f"Wrote final Modal evidence-map summary with {len(rows)} rows to {root}")


if __name__ == "__main__":
    main()
