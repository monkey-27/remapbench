"""Compare ordinary composed-split distributions across two RemapBench sources."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

try:
    from scipy.stats import ks_2samp
except ImportError:
    ks_2samp = None


SPLITS = ["train_composed_seen", "val_composed_seen", "test_composed_seen",
          "test_composed_heldout"]
TUPLES = ["train_tuple_seen", "val_tuple_seen", "test_tuple_seen", "test_tuple_heldout"]
SCALARS = ["value_error", "future_error", "action_error", "action_error_local",
           "path_len_before", "path_len_after", "action_changed_count",
           "action_changed_cell_count"]


def _stats(values):
    x = np.asarray(values, dtype=float)
    return {"mean": float(x.mean()), "std": float(x.std()),
            "p10": float(np.percentile(x, 10)), "p50": float(np.percentile(x, 50)),
            "p90": float(np.percentile(x, 90))}


def _counts(values):
    unique, count = np.unique(values, axis=0, return_counts=True)
    return {str(value.tolist() if getattr(value, "ndim", 0) else int(value)): int(n)
            for value, n in zip(unique, count)}


def _split(path):
    raw = np.load(path, allow_pickle=True)
    d = {key: raw[key] for key in raw.files}
    out = {"n_rows": len(d["intervention_id"]),
           "target_multihot_counts": _counts(d["target_multihot"]),
           "intervention_pair_id_counts": _counts(d.get("intervention_pair_id",
                                                         np.full(len(d["intervention_id"]), -1)))}
    for key in SCALARS:
        if key in d:
            out[key] = _stats(d[key])
    if "path_len_before" in d and "path_len_after" in d:
        out["abs_path_length_change"] = _stats(np.abs(d["path_len_after"] - d["path_len_before"]))
    for key in ["weak_action_change", "action_cell_on_path", "action_cell_near_path"]:
        if key in d:
            out[f"{key}_rate"] = float(np.asarray(d[key], dtype=float).mean())
    return out, d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d1", default="data/remapbench_v1_composed")
    p.add_argument("--d2", default="data/remapbench_cpo_pilot")
    p.add_argument("--out_json", default="results/dataset_comparability_report.json")
    p.add_argument("--out_txt", default="results/dataset_comparability_report.txt")
    args = p.parse_args()
    report = {"d1": args.d1, "d2": args.d2, "scipy_ks_available": ks_2samp is not None,
              "splits": {}, "tuple_splits_d2": {}, "distances": {}}
    raws = {}
    for split in SPLITS:
        report["splits"][split] = {}
        for label, root in [("d1", args.d1), ("d2", args.d2)]:
            summary, raw = _split(os.path.join(root, f"{split}.npz"))
            report["splits"][split][label] = summary
            raws[(split, label)] = raw
        report["distances"][split] = {}
        for key in SCALARS:
            a, b = raws[(split, "d1")], raws[(split, "d2")]
            if key not in a or key not in b:
                continue
            ma, mb = float(np.mean(a[key])), float(np.mean(b[key]))
            row = {"absolute_mean_difference": abs(ma - mb),
                   "ratio_of_means": None if ma == 0 else mb / ma}
            if ks_2samp is not None:
                row["ks_statistic"] = float(ks_2samp(a[key], b[key]).statistic)
            report["distances"][split][key] = row
    for split in TUPLES:
        path = os.path.join(args.d2, f"{split}.npz")
        if os.path.exists(path):
            report["tuple_splits_d2"][split] = _split(path)[0]
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    held = report["distances"]["test_composed_heldout"]
    lines = ["D1 vs D2 dataset comparability", f"scipy KS available: {report['scipy_ks_available']}"]
    for key in ["value_error", "future_error", "action_error", "action_error_local"]:
        lines.append(f"heldout {key}: {held.get(key)}")
    Path(args.out_txt).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
