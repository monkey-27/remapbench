"""Write compact final-fold dataset and evidence controls."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.audit_dataset import audit
from scripts.audit_tuples import audit as audit_tuples
from scripts.final_common import FOLDS, write_json


def _stats(values):
    x = np.asarray(values, dtype=float)
    return {"mean": float(x.mean()), "std": float(x.std()),
            "p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90))}


def _counts(values):
    keys, counts = np.unique(values, axis=0, return_counts=True)
    return {str(key.tolist() if getattr(key, "ndim", 0) else int(key)): int(n)
            for key, n in zip(keys, counts)}


def _component_evidence_rate(d):
    passed, total = 0, 0
    path_change = np.abs(d["path_len_after"] - d["path_len_before"])
    for idx, multihot in enumerate(d["target_multihot"]):
        checks = (
            d["nuisance_error"][idx] >= 0.02,
            d["value_error"][idx] >= 0.02,
            d["future_error"][idx] >= 0.02 or path_change[idx] >= 1 or d["value_error"][idx] >= 0.02,
            d["action_changed_count"][idx] >= 1
            and d["action_changed_cell_count"][idx] >= 1
            and d["action_error_local"][idx] >= 0.20,
        )
        for active, ok in zip(multihot, checks):
            if active:
                total += 1
                passed += int(ok)
    return passed / total if total else None


def _split(path):
    with np.load(path, allow_pickle=True) as raw:
        d = {key: raw[key] for key in raw.files}
    path_change = np.abs(d["path_len_after"] - d["path_len_before"])
    return {
        "n_rows": len(d["intervention_id"]),
        "target_counts": _counts(d["target_multihot"]),
        "pair_counts": _counts(d["intervention_pair_id"]),
        "value_error": _stats(d["value_error"]),
        "future_error": _stats(d["future_error"]),
        "action_error_local": _stats(d["action_error_local"]),
        "weak_action_change_rate": float(np.mean(d["weak_action_change"])),
        "path_length_change": _stats(path_change),
        "action_cell_on_path_rate": float(np.mean(d["action_cell_on_path"])),
        "action_cell_near_path_rate": float(np.mean(d["action_cell_near_path"])),
        "component_evidence_pass_rate": _component_evidence_rate(d),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/final_decomp_gap")
    parser.add_argument("--out-prefix", default="results/final_dataset_evidence_controls")
    args = parser.parse_args()
    report, rows = {"folds": {}}, []
    for fold in FOLDS:
        root = Path(args.data_root) / fold["fold_name"]
        dataset_audit = audit(str(root))
        tuple_report = audit_tuples(str(root))
        split = _split(root / "test_composed_heldout.npz")
        row = {
            "fold_id": fold["fold_id"], "fold_name": fold["fold_name"],
            "heldout_pair_id": fold["heldout_pair_id"],
            "dataset_audit_ok": dataset_audit["pilot_ready"],
            "tuple_audit_ok": tuple_report["ok"],
            "component_evidence_pass_rate": split["component_evidence_pass_rate"],
            "weak_action_change_rate": split["weak_action_change_rate"],
            "value_error_mean": split["value_error"]["mean"],
            "future_error_mean": split["future_error"]["mean"],
            "action_error_local_mean": split["action_error_local"]["mean"],
        }
        rows.append(row)
        report["folds"][fold["fold_name"]] = {**row, "heldout_split": split}
    write_json(f"{args.out_prefix}.json", report)
    Path(f"{args.out_prefix}.csv").parent.mkdir(parents=True, exist_ok=True)
    with open(f"{args.out_prefix}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["Final dataset evidence controls"]
    lines.extend(
        f"{row['fold_name']}: dataset={row['dataset_audit_ok']} tuple={row['tuple_audit_ok']} "
        f"evidence={row['component_evidence_pass_rate']:.3f} weak_action={row['weak_action_change_rate']:.3f}"
        for row in rows)
    Path(f"{args.out_prefix}.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
