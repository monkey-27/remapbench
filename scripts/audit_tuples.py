"""Strict integrity audit for matched RemapBench counterfactual tuples."""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from remapbench.interventions import COMPOSED_PAIRS, INTERVENTION_IDS

TUPLE_SPLITS = [
    "train_tuple_seen", "val_tuple_seen", "test_tuple_seen", "test_tuple_heldout",
]
ROLES = {"base": 0, "A": 1, "B": 2, "AB": 3}
ACTION_LOCAL_MIN = 0.20
REQUIRED_KEYS = {
    "before_grid", "after_grid", "intervention_id", "target_multihot",
    "layout_id", "sample_id", "layout_seed", "intervention_pair_id",
    "tuple_id", "tuple_role", "tuple_role_id", "tuple_pair_id",
    "tuple_component_a", "tuple_component_b", "tuple_replay_order",
    "tuple_b_replay_mask",
}


def _load(path):
    with np.load(path, allow_pickle=True) as raw:
        return {key: raw[key] for key in raw.files}


def _text(value):
    return str(value.item() if isinstance(value, np.ndarray) else value)


def _meaningful_component(data, idx, component):
    ne = float(data["nuisance_error"][idx])
    fe = float(data["future_error"][idx])
    ve = float(data["value_error"][idx])
    pa, pb = int(data["path_len_after"][idx]), int(data["path_len_before"][idx])
    path_change = abs(pa - pb) if pa >= 0 and pb >= 0 else 0
    if component == "sensory_nuisance":
        return ne >= 0.02
    if component == "goal_relocation":
        return ve >= 0.02
    if component == "topology_change":
        return fe >= 0.02 or path_change >= 1 or ve >= 0.02
    if component == "action_change":
        return (
            int(data["action_changed_count"][idx]) >= 1
            and int(data["action_changed_cell_count"][idx]) >= 1
            and float(data["action_error_local"][idx]) >= ACTION_LOCAL_MIN
            and (
                int(data["weak_action_change"][idx]) == 0
                or path_change >= 1
                or ve >= 0.02
                or fe >= 0.02
                or int(data["action_cell_on_path"][idx]) == 1
            )
        )
    return False


def audit(data_dir):
    errors = []
    split_reports = {}
    all_sample_ids = []
    tuple_locations = {}
    heldout_pair_id = 2
    meta_path = os.path.join(data_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
        heldout_pair_id = int(metadata.get("tuple_heldout_pair_id",
                                           metadata.get("heldout_composed_pair_id", 2)))

    for split_name in TUPLE_SPLITS:
        path = os.path.join(data_dir, f"{split_name}.npz")
        if not os.path.exists(path):
            continue
        data = _load(path)
        missing = sorted(REQUIRED_KEYS - set(data))
        if missing:
            errors.append(f"{split_name}: missing keys {missing}")
            continue
        n_rows = len(data["sample_id"])
        if n_rows % 4:
            errors.append(f"{split_name}: row count {n_rows} is not divisible by 4")
        all_sample_ids.extend(int(x) for x in data["sample_id"])
        tuple_ids = sorted(set(int(x) for x in data["tuple_id"]))
        split_reports[split_name] = {"rows": n_rows, "tuples": len(tuple_ids)}

        for tuple_id in tuple_ids:
            idxs = np.where(data["tuple_id"] == tuple_id)[0]
            tuple_locations.setdefault(tuple_id, []).append(split_name)
            if len(idxs) != 4:
                errors.append(f"{split_name}: tuple {tuple_id} has {len(idxs)} rows, expected 4")
                continue
            role_to_idx = {_text(data["tuple_role"][i]): i for i in idxs}
            if set(role_to_idx) != set(ROLES):
                errors.append(f"{split_name}: tuple {tuple_id} roles are {sorted(role_to_idx)}, expected {sorted(ROLES)}")
                continue
            base_idx = role_to_idx["base"]
            layout_ids = {int(data["layout_id"][i]) for i in idxs}
            layout_seeds = {int(data["layout_seed"][i]) for i in idxs}
            sample_ids = {int(data["sample_id"][i]) for i in idxs}
            pair_ids = {int(data["tuple_pair_id"][i]) for i in idxs}
            legacy_pair_ids = {int(data["intervention_pair_id"][i]) for i in idxs}
            components_a = {_text(data["tuple_component_a"][i]) for i in idxs}
            components_b = {_text(data["tuple_component_b"][i]) for i in idxs}
            replay_orders = {_text(data["tuple_replay_order"][i]) for i in idxs}
            if len(layout_ids) != 1 or next(iter(layout_ids)) != tuple_id:
                errors.append(f"{split_name}: tuple {tuple_id} must use one matching layout_id")
            if len(layout_seeds) != 1:
                errors.append(f"{split_name}: tuple {tuple_id} must use one layout_seed")
            if len(sample_ids) != 4:
                errors.append(f"{split_name}: tuple {tuple_id} sample_id values are not distinct")
            if len(pair_ids) != 1 or len(legacy_pair_ids) != 1 or pair_ids != legacy_pair_ids:
                errors.append(f"{split_name}: tuple {tuple_id} has inconsistent pair metadata")
                continue
            pair_id = next(iter(pair_ids))
            if pair_id < 0 or pair_id >= len(COMPOSED_PAIRS):
                errors.append(f"{split_name}: tuple {tuple_id} has invalid pair id {pair_id}")
                continue
            component_a, component_b = COMPOSED_PAIRS[pair_id]
            if components_a != {component_a} or components_b != {component_b}:
                errors.append(f"{split_name}: tuple {tuple_id} component names do not match pair {pair_id}")
            if replay_orders != {"A_then_B"}:
                errors.append(f"{split_name}: tuple {tuple_id} is not marked as sequential A_then_B replay")
            if split_name.endswith("_heldout") and pair_id != heldout_pair_id:
                errors.append(f"{split_name}: tuple {tuple_id} uses pair {pair_id}, expected heldout pair {heldout_pair_id}")
            if split_name.endswith("_seen") and pair_id == heldout_pair_id:
                errors.append(f"{split_name}: tuple {tuple_id} leaks heldout pair {heldout_pair_id}")

            base_before = data["before_grid"][base_idx]
            if not np.array_equal(base_before, data["after_grid"][base_idx]):
                errors.append(f"{split_name}: tuple {tuple_id} base row changes the grid")
            for role, expected_role_id in ROLES.items():
                idx = role_to_idx[role]
                if int(data["tuple_role_id"][idx]) != expected_role_id:
                    errors.append(f"{split_name}: tuple {tuple_id} role {role} has wrong role id")
                if not np.array_equal(data["before_grid"][idx], base_before):
                    errors.append(f"{split_name}: tuple {tuple_id} role {role} does not share the base grid")

            a_idx, b_idx, ab_idx = role_to_idx["A"], role_to_idx["B"], role_to_idx["AB"]
            expected_ids = {
                "base": -1,
                "A": INTERVENTION_IDS[component_a],
                "B": INTERVENTION_IDS[component_b],
                "AB": INTERVENTION_IDS["composed"],
            }
            for role, expected_id in expected_ids.items():
                if int(data["intervention_id"][role_to_idx[role]]) != expected_id:
                    errors.append(f"{split_name}: tuple {tuple_id} role {role} has wrong intervention_id")
            if np.any(data["target_multihot"][base_idx]):
                errors.append(f"{split_name}: tuple {tuple_id} base target must be zero")
            expected_ab = np.maximum(data["target_multihot"][a_idx], data["target_multihot"][b_idx])
            if not np.array_equal(data["target_multihot"][ab_idx], expected_ab):
                errors.append(f"{split_name}: tuple {tuple_id} AB target is not the union of A and B")
            for idx, component in ((a_idx, component_a), (b_idx, component_b),
                                   (ab_idx, component_a), (ab_idx, component_b)):
                if not _meaningful_component(data, idx, component):
                    role = _text(data["tuple_role"][idx])
                    errors.append(
                        f"{split_name}: tuple {tuple_id} role {role} lacks meaningful {component} evidence")
            expected_grid_ab = base_before.copy()
            changed_a = data["after_grid"][a_idx] != base_before
            changed_b = data["tuple_b_replay_mask"][ab_idx].astype(bool)
            expected_grid_ab[changed_a] = data["after_grid"][a_idx][changed_a]
            expected_grid_ab[changed_b] = data["after_grid"][b_idx][changed_b]
            if not np.array_equal(data["after_grid"][ab_idx], expected_grid_ab):
                errors.append(f"{split_name}: tuple {tuple_id} AB grid is not the merged A+B counterfactual")

    if len(all_sample_ids) != len(set(all_sample_ids)):
        errors.append("duplicate sample_id values across tuple splits")
    for tuple_id, locations in tuple_locations.items():
        if len(locations) != 1:
            errors.append(f"tuple_id {tuple_id} appears in multiple splits: {locations}")
    if not split_reports:
        errors.append("no tuple splits found")

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
        for split_name, requested in metadata.get("requested_tuple_counts", {}).items():
            actual = split_reports.get(split_name, {}).get("tuples", 0)
            if actual != int(requested):
                errors.append(f"{split_name}: saved {actual} tuples, requested {requested}")

    tuple_sample_ids = set(all_sample_ids)
    tuple_layout_ids = set(tuple_locations)
    for path in glob.glob(os.path.join(data_dir, "*.npz")):
        split_name = os.path.splitext(os.path.basename(path))[0]
        if split_name in TUPLE_SPLITS:
            continue
        data = _load(path)
        if "sample_id" not in data:
            continue
        overlap = tuple_sample_ids & {int(x) for x in data["sample_id"]}
        if overlap:
            errors.append(f"{split_name}: {len(overlap)} sample_id values overlap tuple splits")
        if "layout_id" in data:
            layout_overlap = tuple_layout_ids & {int(x) for x in data["layout_id"]}
            if layout_overlap:
                errors.append(f"{split_name}: {len(layout_overlap)} layout_id values overlap tuple splits")

    report = {
        "data_dir": data_dir,
        "splits": split_reports,
        "heldout_pair_id": heldout_pair_id,
        "ok": not errors,
        "errors": errors,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out")
    parser.add_argument("--strict", action="store_true",
                        help="Exit nonzero when tuple integrity checks fail")
    args = parser.parse_args()
    report = audit(args.data_dir)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
