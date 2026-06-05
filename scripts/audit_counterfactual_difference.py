"""Audit tuple and primitive-mask assumptions for evidence-mask training."""
import argparse
import json
import os
import sys
import zipfile
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from remapbench.counterfactual_difference import (
    CAUSES, PRIMITIVE_CHANNELS, ROLE_TO_ID, CounterfactualTransitionDataset, split_path,
)
from scripts.evaluate import compute_multilabel_metrics
from scripts.final_common import load_config, write_json


REQUIRED_TUPLE_KEYS = {
    "tuple_id", "tuple_role", "tuple_role_id", "tuple_pair_id",
    "tuple_component_a", "tuple_component_b", "tuple_replay_order",
}
ID_TO_ROLE = {value: key for key, value in ROLE_TO_ID.items()}


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


def primitive_diff_rule(path):
    """Rule baseline from primitive channel diffs, not an oracle for target labels."""
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


def _as_int(value):
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _bits(values):
    return "".join(str(int(x)) for x in values)


def primitive_diff_rule_disagreement(path, split, max_examples=20):
    ds = CounterfactualTransitionDataset(path)
    by_cause = {cause: Counter() for cause in CAUSES}
    by_role = defaultdict(Counter)
    by_pair = defaultdict(Counter)
    subset_confusion = Counter()
    examples = []
    for idx in range(len(ds)):
        item = ds[idx]
        target = item["target_multihot"].numpy().astype(int)
        masks = item["primitive_masks"]
        mask_sums = masks.flatten(1).sum(dim=1).numpy()
        pred = (mask_sums > 0).astype(int)
        raw_diff = (item["after_grid"] - item["before_grid"]).abs()
        channel_group_sums = {
            CAUSES[cause_idx]: float(raw_diff[list(channels)].sum().item())
            for cause_idx, channels in PRIMITIVE_CHANNELS.items()
        }
        role_id = _as_int(item["transition_role_id"])
        role = ID_TO_ROLE.get(role_id, str(role_id))
        pair_id = _as_int(item["pair_id"])
        subset_confusion[f"{_bits(target)}->{_bits(pred)}"] += 1
        for cause_idx, cause in enumerate(CAUSES):
            active_mask_zero = int(target[cause_idx] == 1 and pred[cause_idx] == 0)
            inactive_mask_active = int(target[cause_idx] == 0 and pred[cause_idx] == 1)
            by_cause[cause]["n"] += 1
            by_cause[cause]["target_active_mask_inactive"] += active_mask_zero
            by_cause[cause]["target_inactive_mask_active"] += inactive_mask_active
            by_cause[cause]["active_mask_nonzero"] += int(target[cause_idx] == 1 and pred[cause_idx] == 1)
            by_cause[cause]["inactive_mask_nonzero"] += inactive_mask_active
            by_role[role][f"{cause}_target_active_mask_inactive"] += active_mask_zero
            by_role[role][f"{cause}_target_inactive_mask_active"] += inactive_mask_active
            by_pair[str(pair_id)][f"{cause}_target_active_mask_inactive"] += active_mask_zero
            by_pair[str(pair_id)][f"{cause}_target_inactive_mask_active"] += inactive_mask_active
        if (target != pred).any() and len(examples) < max_examples:
            examples.append({
                "row_index": idx,
                "sample_id": _as_int(item["sample_id"]),
                "tuple_id": _as_int(item["tuple_id"]),
                "pair_id": pair_id,
                "transition_role": role,
                "target_multihot": target.tolist(),
                "primitive_diff_rule_prediction": pred.tolist(),
                "primitive_mask_sums": {cause: float(mask_sums[i]) for i, cause in enumerate(CAUSES)},
                "channel_group_raw_diff_sums": channel_group_sums,
            })
    metrics = primitive_diff_rule(path)["metrics"]
    return {
        "split": split,
        "n_transitions": len(ds),
        "primitive_diff_rule": metrics,
        "subset_confusion": dict(subset_confusion),
        "by_cause": {cause: dict(counter) for cause, counter in by_cause.items()},
        "by_transition_role": {role: dict(counter) for role, counter in by_role.items()},
        "by_pair_id": {pair: dict(counter) for pair, counter in by_pair.items()},
        "failure_examples": examples,
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
        split: primitive_diff_rule(split_path(data_dir, split))
        for split in ("train_tuple_seen", "test_tuple_heldout", "test_composed_heldout")
        if os.path.exists(split_path(data_dir, split))
    }
    disagreement = {}
    failure_examples = {}
    for split in ("train_tuple_seen", "val_tuple_seen", "test_tuple_seen",
                  "test_tuple_heldout", "test_single", "test_composed_seen",
                  "test_composed_heldout"):
        path = split_path(data_dir, split)
        if os.path.exists(path):
            try:
                row = primitive_diff_rule_disagreement(path, split)
                disagreement[split] = {key: value for key, value in row.items()
                                       if key != "failure_examples"}
                failure_examples[split] = row["failure_examples"]
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                disagreement[split] = {"status": "failed", "error": repr(exc)}
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
        "primitive_diff_rule": primitive,
        "primitive_diff_rule_disagreement": disagreement,
        "label_balance": label_balance,
    }
    out = args.out or os.path.join("results", cfg["run_name"], "audit_counterfactual_difference.json")
    write_json(out, report)
    run_dir = os.path.dirname(out)
    write_json(os.path.join(run_dir, "primitive_diff_rule_audit.json"), {
        "note": "primitive_diff_rule is a rule baseline from primitive channel diffs, not an oracle for target labels.",
        "splits": disagreement,
    })
    write_json(os.path.join(run_dir, "primitive_diff_rule_failure_examples.json"), failure_examples)
    print(f"Evidence-mask audit {report['status']} errors={len(errors)} warnings={len(report['warnings'])} -> {out}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
