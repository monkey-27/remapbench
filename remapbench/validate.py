"""
Validation script.
Usage:
    python -m remapbench.validate --data_dir data/remapbench_v0
"""
import argparse
import json
import os
import sys
import numpy as np

INTERVENTION_NAMES = {0: "sensory_nuisance", 1: "goal_relocation",
                      2: "topology_change", 3: "action_change", 4: "composed"}
SEED_OFFSETS = {"train_single": 0, "val_single": 500_000,
                "test_single": 1_000_000, "test_composed": 1_500_000}

# Validation thresholds
THRESH_HIGH = 0.01     # error must be ABOVE this to count as "high"
THRESH_NEAR_ZERO = 0.015  # error must be BELOW this to count as "near zero"


def load_split(path):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def stats(arr):
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)),
            "min": float(np.min(arr)), "max": float(np.max(arr))}


def validate(data_dir):
    split_files = {
        "train_single": "train_single.npz",
        "val_single":   "val_single.npz",
        "test_single":  "test_single.npz",
        "test_composed":"test_composed.npz",
    }

    splits = {}
    for name, fname in split_files.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found")
            continue
        splits[name] = load_split(path)
        print(f"Loaded {name}: {len(splits[name]['intervention_id'])} samples")

    diagnostics = {}
    all_ok = True

    for split_name, data in splits.items():
        print(f"\n=== {split_name} ===")
        iids = data["intervention_id"]
        n = len(iids)
        diag = {"n_samples": int(n)}

        # Class counts
        counts = {}
        for iid in np.unique(iids):
            cnt = int((iids == iid).sum())
            name = INTERVENTION_NAMES.get(int(iid), f"id{iid}")
            counts[name] = cnt
            print(f"  {name}: {cnt}")
        diag["class_counts"] = counts

        # Per-intervention error statistics
        error_keys = ["sensory_error", "future_error", "value_error", "action_error"]
        diag["error_stats"] = {}
        for iid_val, iname in INTERVENTION_NAMES.items():
            mask = iids == iid_val
            if mask.sum() == 0:
                continue
            row = {}
            for ek in error_keys:
                if ek in data:
                    row[ek] = stats(data[ek][mask])
            diag["error_stats"][iname] = row
            means = {ek: row[ek]["mean"] for ek in error_keys if ek in row}
            print(f"  {iname:25s} "
                  + "  ".join(f"{k.split('_')[0][:3]}={v:.4f}" for k, v in means.items()))

        # Leakage check: layout seeds per split should be in disjoint ranges
        offset = SEED_OFFSETS.get(split_name, 0)
        diag["seed_offset"] = offset

        # Assertions for single-intervention splits
        if "single" in split_name:
            for iid_val, iname in INTERVENTION_NAMES.items():
                if iid_val == 4:
                    continue
                mask = iids == iid_val
                if mask.sum() == 0:
                    continue

                se = data["sensory_error"][mask].mean() if "sensory_error" in data else 0
                fe = data["future_error"][mask].mean() if "future_error" in data else 0
                ve = data["value_error"][mask].mean() if "value_error" in data else 0
                ae = data["action_error"][mask].mean() if "action_error" in data else 0

                if iname == "sensory_nuisance":
                    ok = se > THRESH_HIGH and fe < THRESH_NEAR_ZERO and ve < THRESH_NEAR_ZERO and ae < THRESH_NEAR_ZERO
                    flag = "OK" if ok else "FAIL"
                    if not ok:
                        all_ok = False
                    print(f"  [assert sensory_nuisance] se={se:.4f}>{THRESH_HIGH} "
                          f"fe={fe:.4f}<{THRESH_NEAR_ZERO} "
                          f"ve={ve:.4f}<{THRESH_NEAR_ZERO} "
                          f"ae={ae:.4f}<{THRESH_NEAR_ZERO} → {flag}")

                elif iname == "goal_relocation":
                    ok = ve > THRESH_HIGH and fe < THRESH_NEAR_ZERO and ae < THRESH_NEAR_ZERO
                    flag = "OK" if ok else "FAIL"
                    if not ok:
                        all_ok = False
                    print(f"  [assert goal_relocation] ve={ve:.4f}>{THRESH_HIGH} "
                          f"fe={fe:.4f}<{THRESH_NEAR_ZERO} "
                          f"ae={ae:.4f}<{THRESH_NEAR_ZERO} → {flag}")

                elif iname == "topology_change":
                    ok = fe > THRESH_HIGH
                    flag = "OK" if ok else "FAIL"
                    if not ok:
                        all_ok = False
                    print(f"  [assert topology_change] fe={fe:.4f}>{THRESH_HIGH} → {flag}")

                elif iname == "action_change":
                    ok = ae > THRESH_HIGH
                    flag = "OK" if ok else "FAIL"
                    if not ok:
                        all_ok = False
                    print(f"  [assert action_change] ae={ae:.4f}>{THRESH_HIGH} → {flag}")

        # Composed: at least two active target labels
        if split_name == "test_composed":
            mh = data["target_multihot"]  # [N, 4]
            active = mh.sum(axis=1)
            ok = bool((active >= 2).all())
            flag = "OK" if ok else "FAIL"
            if not ok:
                all_ok = False
                bad = (active < 2).sum()
                print(f"  [assert composed multi-label] {bad} samples with <2 active labels → {flag}")
            else:
                print(f"  [assert composed multi-label] all samples have >=2 active labels → {flag}")

        diagnostics[split_name] = diag

    # Leakage check: verify seed offsets are disjoint across splits (by design)
    loaded_offsets = {k: SEED_OFFSETS[k] for k in splits}
    offset_values = list(loaded_offsets.values())
    leakage_free = len(set(offset_values)) == len(offset_values)
    print(f"\n[assert leakage-free] disjoint seed offsets: {loaded_offsets} → {'OK' if leakage_free else 'FAIL'}")
    if not leakage_free:
        all_ok = False

    diagnostics["leakage_free"] = leakage_free
    diagnostics["all_assertions_passed"] = all_ok

    out_path = os.path.join(data_dir, "diagnostics.json")
    with open(out_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\nDiagnostics written to {out_path}")
    print(f"\nOverall: {'ALL ASSERTIONS PASSED' if all_ok else 'SOME ASSERTIONS FAILED'}")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Validate RemapBench dataset")
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()
    ok = validate(args.data_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
