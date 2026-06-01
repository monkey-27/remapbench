"""Aggregate final paper artifacts while explicitly retaining missing runs."""
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


def _read(path):
    return json.loads(path.read_text()) if path.exists() else None


def _mean(rows, key):
    values = [row[key] for row in rows if row and row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _std(rows, key):
    values = [row[key] for row in rows if row and row.get(key) is not None]
    return float(np.std(values)) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--figures-root", default="figures/final")
    args = parser.parse_args()
    root, figures = Path(args.results_root), Path(args.figures_root)
    root.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    direct, oracle, patching, seen = [], [], [], []
    missing = []
    for config_path, cfg in configs():
        run = root / cfg["run_name"]
        if "final_direct_" in cfg["run_name"] or "final_seen_upper_bound_" in cfg["run_name"]:
            report = _read(run / "eval_final.json")
            if report is None:
                missing.append(str(run / "eval_final.json"))
                continue
            control_split = (
                "test_composed_seen" if "seen_upper_bound" in cfg["run_name"]
                else "test_composed_heldout")
            row = {**report, **report["split_metrics"][control_split],
                   "reported_split": control_split}
            (seen if "seen_upper_bound" in cfg["run_name"] else direct).append(row)
            if (cfg["model"] in ("gated_erpm", "factorized_gates")
                    and "seen_upper_bound" not in cfg["run_name"]):
                patch = _read(run / "causal_patching.json")
                if patch is None:
                    missing.append(str(run / "causal_patching.json"))
                else:
                    patching.append(patch)
        elif "final_tuple_" in cfg["run_name"]:
            report = _read(run / "oracle_tuple_gap.json")
            if report is None:
                missing.append(str(run / "oracle_tuple_gap.json"))
            else:
                oracle.append(report)
    controls = _read(root / "final_dataset_evidence_controls.json")

    direct_table = []
    for model in ("standard", "gated", "independent_gate_heads"):
        rows = [row for row in direct if row["report_model_name"] == model]
        direct_table.append({
            "model": model, "n_runs": len(rows), "mean_exact": _mean(rows, "exact_match"),
            "std_exact": _std(rows, "exact_match"), "mean_macro_f1": _mean(rows, "macro_f1"),
            "miss_sensory": _mean(rows, "miss_sensory"), "miss_value": _mean(rows, "miss_value"),
            "miss_map": _mean(rows, "miss_map"), "miss_action": _mean(rows, "miss_action"),
        })
    summary = {
        "status": "complete" if not missing else "incomplete",
        "missing_expected_reports": missing,
        "table1_direct_heldout_failure": direct_table,
        "table2_direct_vs_oracle_tuple_gap": oracle,
        "table3_causal_patching": patching,
        "table4_controls": {"seen_pair_upper_bound": seen, "dataset_evidence": controls},
    }
    (root / "final_paper_summary.json").write_text(json.dumps(summary, indent=2))
    with (root / "final_paper_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(direct_table[0]))
        writer.writeheader()
        writer.writerows(direct_table)
    lines = ["Final decomposition-gap paper tables", "", "Table 1 - Direct heldout failure"]
    lines.extend(json.dumps(row, sort_keys=True) for row in direct_table)
    lines += ["", "Table 2 - Direct vs oracle tuple gap"]
    lines.extend(json.dumps(row, sort_keys=True) for row in oracle)
    lines += ["", "Table 3 - Causal patching"]
    lines.extend(json.dumps(row, sort_keys=True) for row in patching)
    lines += ["", "Table 4 - Controls", json.dumps(summary["table4_controls"], sort_keys=True)]
    lines += ["", f"Missing expected reports: {len(missing)}"]
    lines.extend(missing)
    (root / "final_paper_tables.txt").write_text("\n".join(lines) + "\n")

    def _bar(path, labels, values, ylabel):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, [0 if value is None else value for value in values])
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(figures / path, dpi=160)
        plt.close(fig)

    _bar("direct_heldout_exact.png", [row["model"] for row in direct_table],
         [row["mean_exact"] for row in direct_table], "Mean heldout exact")
    _bar("tuple_gap.png", [str(row.get("fold_name")) for row in oracle],
         [row.get("tuple_direct_gap_exact") for row in oracle], "Oracle tuple - direct exact")
    _bar("causal_patching_rescue.png", [str(row.get("report_model_name")) for row in patching],
         [row.get("correct_patch_rescue_rate_among_failed") for row in patching], "Rescue rate")
    _bar("control_summary.png", [str(row.get("report_model_name")) for row in seen],
         [row.get("exact_match") for row in seen], "Seen-pair exact")
    print(f"Aggregation status: {summary['status']} ({len(missing)} missing expected reports)")


if __name__ == "__main__":
    main()
