"""Aggregate local CDT pilot outputs."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.final_common import write_json


BASELINES = {
    "standard_cnn_direct_mean": 0.1971,
    "shared_gate_direct_mean": 0.1024,
    "independent_gate_heads_direct_mean": 0.0866,
}


def _read(path):
    return json.loads(path.read_text()) if path.exists() else None


def row_for(run_dir):
    report = _read(run_dir / "eval_counterfactual_difference.json")
    audit = _read(run_dir / "audit_counterfactual_difference.json")
    if not report:
        return None
    heldout = report["split_metrics"].get("test_composed_heldout", {})
    tuple_metrics = report["split_metrics"].get("test_tuple_heldout_transitions", {})
    oracle = report["primitive_oracle"].get("test_composed_heldout", {})
    context = report.get("context_diagnostics", {})
    return {
        "run_name": run_dir.name,
        "fold_name": report.get("fold_name"),
        "model": report.get("report_model_name"),
        "seed": report.get("seed"),
        "audit_status": audit.get("status") if audit else None,
        "audit_warnings": len(audit.get("warnings", [])) if audit else None,
        "heldout_exact": heldout.get("exact_match"),
        "heldout_macro_f1": heldout.get("macro_f1"),
        "tuple_transition_exact": tuple_metrics.get("exact_match"),
        "primitive_oracle_exact": oracle.get("exact_match"),
        "inactive_evidence": heldout.get("inactive_evidence_magnitude"),
        "symmetric_ba_available": context.get("symmetric_ba_available"),
        "context_gap_by_cause": context.get("context_gap_by_cause"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args()
    root = Path(args.results_root)
    rows = [row for row in (row_for(path) for path in sorted(root.glob("cdt_*"))) if row]
    trusted = [row for row in rows if row["audit_status"] in (None, "pass")]
    summary = {
        "status": "complete",
        "baseline_comparison_constants": BASELINES,
        "rows": rows,
        "trusted_rows": trusted,
        "decision_note": (
            "Promising local pilot if high heldout rows have audit_status pass; "
            "not fully convincing until BA symmetry and all four audited folds are available."
        ),
    }
    write_json(root / "cdt_pilot_summary.json", summary)
    csv_path = root / "cdt_pilot_summary.csv"
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["Counterfactual difference pilot", "", "Baselines", json.dumps(BASELINES, sort_keys=True), "", "Runs"]
    lines.extend(json.dumps(row, sort_keys=True) for row in rows)
    (root / "cdt_pilot_tables.txt").write_text("\n".join(lines) + "\n")
    print(f"Aggregated {len(rows)} CDT runs -> {root / 'cdt_pilot_summary.json'}")


if __name__ == "__main__":
    main()
