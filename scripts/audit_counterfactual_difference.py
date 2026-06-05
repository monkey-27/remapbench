"""Audit tuple and primitive-mask assumptions for CDT."""
import argparse
import json
import os
import sys
import zipfile
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from remapbench.counterfactual_difference import CAUSES, CounterfactualTransitionDataset, split_path
from scripts.evaluate import compute_multilabel_metrics
from scripts.final_common import load_config, write_json


REQUIRED_TUPLE_KEYS = {
    "tuple_id", "tuple_role", "tuple_role_id", "tuple_pair_id",
    "tuple_component_a", "tuple_component_b", "tuple_replay_order",
}


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _split_ids(path):
    try:
        with np.load(path, allow_pickle=True) as data:
            return {
                "layout_id": set(map(int, data["layout_id"])) if "layout_id" in data else set(),
                "sample_id": set(map(int, data["sample_id"])) if "sample_id" in data else set(),
                "tuple_id": set(map(int, data["tuple_id"])) if "tuple_id" in data else set(),
                "error": None,
            }
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        return {"layout_id": set(), "sample_id": set(), "tuple_id": set(), "error": repr(exc)}


def primitive_oracle(path):
    ds = CounterfactualTransitionDataset(path)
    preds, targets = [], []
    role_counts = Counter()
    mask_active = defaultdict(list)
    inactive_leak = defaultdict(list)
    for item in ds:
        pred = (item["primitive_masks"].flatten(1).sum(dim=1) > 0).float().numpy()
        target = item["target_multihot"].numpy()
        preds.append(pred)
        targets.append(target)
        role_counts[int(item["transition_role_id"])] += 1
        mask_sum = item["primitive_masks"].flatten(1).sum(dim=1).numpy()
        for i, cause in enumerate(CAUSES):
            if target[i] > 0.5:
                mask_active[cause].append(float(mask_sum[i] > 0))
            else:
                inactive_leak[cause].append(float(mask_sum[i] > 0))
    return {
        "metrics": compute_multilabel_metrics(np.stack(preds), np.stack(targets)),
        "role_counts": dict(role_counts),
        "active_mask_nonzero_rate": {k: float(np.mean(v)) if v else None for k, v in mask_active.items()},
        "inactive_mask_nonzero_rate": {k: float(np.mean(v)) if v else None for k, v in inactive_leak.items()},
    }


def audit_tuple_file(path, heldout_pair_id, expect_seen):
    errors, warnings = [], []
    with np.load(path, allow_pickle=True) as data:
        missing = sorted(REQUIRED_TUPLE_KEYS - set(data.files))
        if missing:
            errors.append(f"{path}: missing tuple keys {missing}")
            return {"errors": errors, "warnings": warnings}
        roles_available = {_text(x) for x in data["tuple_role"]}
        if "BA" not in roles_available:
            warnings.append(f"{path}: BA role is absent; CDT context loss is A_then_B only")
        pair_counts = Counter(map(int, data["tuple_pair_id"]))
        role_counts = Counter(_text(x) for x in data["tuple_role"])
        for tuple_id in sorted(set(map(int, data["tuple_id"]))):
            idxs = np.where(data["tuple_id"] == tuple_id)[0]
            roles = {_text(data["tuple_role"][i]) for i in idxs}
            if roles != {"base", "A", "B", "AB"}:
                errors.append(f"{path}: tuple {tuple_id} roles {sorted(roles)} != base/A/B/AB")
            pair_ids = {int(data["tuple_pair_id"][i]) for i in idxs}
            if len(pair_ids) != 1:
                errors.append(f"{path}: tuple {tuple_id} has multiple pair ids {sorted(pair_ids)}")
            pair_id = next(iter(pair_ids))
            if expect_seen and pair_id == heldout_pair_id:
                errors.append(f"{path}: seen tuple {tuple_id} leaks heldout pair {heldout_pair_id}")
            if not expect_seen and pair_id != heldout_pair_id:
                errors.append(f"{path}: heldout tuple {tuple_id} uses pair {pair_id}, expected {heldout_pair_id}")
        return {
            "errors": errors,
            "warnings": warnings,
            "pair_counts": dict(pair_counts),
            "role_counts": dict(role_counts),
            "n_tuples": len(set(map(int, data["tuple_id"]))),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_dir = cfg["data_dir"]
    heldout = int(cfg["heldout_composed_pair_id"])
    errors, warnings = [], []
    tuple_reports = {}
    for split, expect_seen in (("train_tuple_seen", True), ("val_tuple_seen", True), ("test_tuple_seen", True), ("test_tuple_heldout", False)):
        path = split_path(data_dir, split)
        if os.path.exists(path):
            report = audit_tuple_file(path, heldout, expect_seen)
            tuple_reports[split] = report
            errors.extend(report.get("errors", []))
            warnings.extend(report.get("warnings", []))
        else:
            warnings.append(f"{path}: tuple split missing")
    ids = {split: _split_ids(split_path(data_dir, split))
           for split in ("train_single", "val_single", "test_composed_heldout")
           if os.path.exists(split_path(data_dir, split))}
    for split, row in ids.items():
        if row.get("error"):
            errors.append(f"{split_path(data_dir, split)} failed to load: {row['error']}")
    leakage = {}
    for left in ids:
        for right in ids:
            if left >= right:
                continue
            overlap = ids[left]["layout_id"] & ids[right]["layout_id"]
            leakage[f"{left}__{right}"] = len(overlap)
            if overlap:
                errors.append(f"layout leakage {left}/{right}: {len(overlap)} overlaps")
    primitive = {
        split: primitive_oracle(split_path(data_dir, split))
        for split in ("train_tuple_seen", "test_tuple_heldout", "test_composed_heldout")
        if os.path.exists(split_path(data_dir, split))
    }
    label_balance = {}
    for split in ("train_tuple_seen", "test_tuple_heldout"):
        path = split_path(data_dir, split)
        if os.path.exists(path):
            with np.load(path, allow_pickle=True) as data:
                label_balance[split] = np.asarray(data["target_multihot"]).sum(axis=0).astype(float).tolist()
    report = {
        "status": "pass" if not errors else "fail",
        "fully_symmetric_ba": False,
        "errors": errors,
        "warnings": sorted(set(warnings)),
        "tuple_reports": tuple_reports,
        "split_layout_overlap_counts": leakage,
        "primitive_oracle": primitive,
        "label_balance": label_balance,
    }
    out = args.out or os.path.join("results", cfg["run_name"], "audit_counterfactual_difference.json")
    write_json(out, report)
    print(f"CDT audit {report['status']} errors={len(errors)} warnings={len(report['warnings'])} -> {out}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
