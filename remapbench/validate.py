"""
Validation script.
Usage:
    python -m remapbench.validate --data_dir data/remapbench_v0

Reports class counts, per-sample pass rates, layout_id leakage check,
NaN/Inf checks, and weak_action_change rate. Writes diagnostics.json.
"""
import argparse
import json
import os
import sys
import numpy as np

INTERVENTION_NAMES = {
    0: "sensory_nuisance", 1: "goal_relocation",
    2: "topology_change",  3: "action_change", 4: "composed",
}
TARGET_NAMES = ["sensory_update", "value_remap", "map_remap", "action_remap"]

# Per-sample pass thresholds
THRESH_HIGH       = 0.01
THRESH_NEAR_ZERO  = 0.015
MIN_PASS_RATE     = 0.90
WEAK_ACTION_WARN  = 0.10   # warn if weak rate exceeds this
WEAK_ACTION_FAIL  = 0.25   # hard fail if weak rate exceeds this
ACTION_LOCAL_MIN  = 0.20


def load_split(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _stats(arr):
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)),
            "min": float(np.min(arr)), "max": float(np.max(arr))}


def _check_sample_sensory(s):
    return (float(s["nuisance_error"]) > THRESH_HIGH and
            float(s["future_error"])   < THRESH_NEAR_ZERO and
            float(s["value_error"])    < THRESH_NEAR_ZERO and
            float(s["action_error"])   < THRESH_NEAR_ZERO)


def _check_sample_goal(s):
    return (float(s["value_error"])  > THRESH_HIGH and
            float(s["future_error"]) < THRESH_NEAR_ZERO and
            float(s["action_error"]) < THRESH_NEAR_ZERO)


def _check_sample_topology(s):
    return float(s["future_error"]) > THRESH_HIGH


def _check_sample_action(s):
    if all(k in s for k in ("action_changed_count", "action_changed_cell_count",
                            "action_error_local")):
        return (int(s["action_changed_count"]) >= 1 and
                int(s["action_changed_cell_count"]) >= 1 and
                float(s["action_error_local"]) >= ACTION_LOCAL_MIN)
    return float(s["action_error"]) > THRESH_HIGH


def _check_sample_composed(s):
    mh = s["target_multihot"]
    return int(mh.sum()) >= 2


_CHECKERS = {
    0: _check_sample_sensory,
    1: _check_sample_goal,
    2: _check_sample_topology,
    3: _check_sample_action,
    4: _check_sample_composed,
}


def _sample_at(data, i):
    """Return a per-row dict view of a stacked-array data dict."""
    return {k: v[i] for k, v in data.items()}


def validate(data_dir):
    # Core splits (always expected); optional robustness splits loaded if present.
    required_files = {
        "train_single":  "train_single.npz",
        "val_single":    "val_single.npz",
        "test_single":   "test_single.npz",
        "test_composed": "test_composed.npz",
    }
    optional_files = {
        "test_larger": "test_larger.npz",
        "test_noisy":  "test_noisy.npz",
        "train_composed_seen":   "train_composed_seen.npz",
        "val_composed_seen":     "val_composed_seen.npz",
        "test_composed_seen":    "test_composed_seen.npz",
        "test_composed_heldout": "test_composed_heldout.npz",
    }

    splits = {}
    for name, fname in {**required_files, **optional_files}.items():
        path = os.path.join(data_dir, fname)
        is_optional = name in optional_files
        if os.path.exists(path):
            splits[name] = load_split(path)
            print(f"Loaded {name}: {len(splits[name]['intervention_id'])} samples")
        elif is_optional:
            print(f"  [OPTIONAL] {path} not found — skipping")
        else:
            print(f"  [SKIP] {path} not found")

    diagnostics = {}
    all_ok = True

    # -----------------------------------------------------------------------
    # Requested vs actual split sizes (from metadata.json, if present)
    # -----------------------------------------------------------------------
    meta_path = os.path.join(data_dir, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            meta = None
        if meta and isinstance(meta.get("requested_split_sizes"), dict):
            print("\n--- Requested vs actual split sizes ---")
            req = meta["requested_split_sizes"]
            for name, requested_n in req.items():
                requested_n = int(requested_n)
                if requested_n <= 0:
                    continue
                if name not in splits and "_tuple_" in name:
                    print(f"  {name:15s} handled by scripts/audit_tuples.py")
                    continue
                actual_n = len(splits[name]["intervention_id"]) if name in splits else 0
                frac = actual_n / requested_n if requested_n else float("nan")
                print(f"  {name:15s} {actual_n:>6d}/{requested_n:<6d} (fraction {frac:.3f})")
            print("  (audit_dataset.py is the strict gate that hard-fails on low completion)")

    # -----------------------------------------------------------------------
    # NaN / Inf check
    # -----------------------------------------------------------------------
    print("\n--- NaN/Inf checks ---")
    for split_name, data in splits.items():
        float_keys = ["delta_future", "delta_value", "action_delta",
                      "future_before", "future_after", "value_before", "value_after",
                      "nuisance_error", "full_visual_error", "future_error",
                      "value_error", "action_error"]
        bad = []
        for k in float_keys:
            if k not in data:
                continue
            arr = data[k].astype(np.float32)
            if not np.isfinite(arr).all():
                bad.append(k)
        flag = "OK" if not bad else f"FAIL: {bad}"
        print(f"  {split_name}: {flag}")
        if bad:
            all_ok = False

    # -----------------------------------------------------------------------
    # Layout ID leakage check
    # -----------------------------------------------------------------------
    print("\n--- Layout ID leakage ---")
    layout_id_sets = {}
    for split_name, data in splits.items():
        if "layout_id" in data:
            layout_id_sets[split_name] = set(data["layout_id"].tolist())
        else:
            layout_id_sets[split_name] = set()

    leakage_free = True
    split_names = list(layout_id_sets.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            sa, sb = split_names[i], split_names[j]
            overlap = layout_id_sets[sa] & layout_id_sets[sb]
            if overlap:
                print(f"  FAIL: {sa} ∩ {sb} = {len(overlap)} shared layout IDs")
                leakage_free = False
            else:
                print(f"  OK: {sa} ∩ {sb} = empty")
    if leakage_free:
        print("  All layout IDs are disjoint across splits.")
    diagnostics["leakage_free"] = leakage_free
    if not leakage_free:
        all_ok = False

    # -----------------------------------------------------------------------
    # Per-split analysis
    # -----------------------------------------------------------------------
    for split_name, data in splits.items():
        print(f"\n=== {split_name} ===")
        iids = data["intervention_id"]
        n = len(iids)
        diag = {"n_samples": int(n), "class_counts": {}, "error_stats": {},
                "pass_rates": {}, "target_label_counts": {}}

        # Class counts
        for iid in sorted(np.unique(iids)):
            cnt = int((iids == iid).sum())
            name = INTERVENTION_NAMES.get(int(iid), f"id{iid}")
            diag["class_counts"][name] = cnt
            print(f"  {name}: {cnt}")

        # target_multihot validation
        mh = data["target_multihot"]
        assert mh.shape[1] == 4, f"target_multihot shape mismatch: {mh.shape}"
        for i, lname in enumerate(TARGET_NAMES):
            cnt = int(mh[:, i].sum())
            diag["target_label_counts"][lname] = cnt
        print(f"  label counts: {diag['target_label_counts']}")

        # Composed pair counts (structural view only; audit checks evidence)
        if "intervention_pair_id" in data:
            pid = data["intervention_pair_id"]
            uniq = sorted(set(int(x) for x in pid.tolist() if int(x) >= 0))
            if uniq:
                pair_counts = {int(u): int((pid == u).sum()) for u in uniq}
                diag["composed_pair_counts"] = pair_counts
                print(f"  composed pair counts: {pair_counts}")

        is_single   = "single" in split_name or split_name in ("test_larger", "test_noisy")
        is_composed = "composed" in split_name

        if is_single:
            sums = mh.sum(axis=1)
            n_bad = int((sums != 1).sum())
            if n_bad > 0:
                print(f"  [WARN] {n_bad} single samples with target_sum != 1")

        if is_composed:
            sums = mh.sum(axis=1)
            n_bad = int((sums < 2).sum())
            ok = n_bad == 0
            print(f"  [assert composed ≥2 labels]: {'OK' if ok else f'FAIL ({n_bad} samples < 2 active)'}")
            if not ok:
                all_ok = False

        # Per-intervention error stats + pass rates
        for iid_val in sorted(np.unique(iids)):
            mask = iids == iid_val
            if mask.sum() == 0:
                continue
            iname = INTERVENTION_NAMES.get(int(iid_val), f"id{iid_val}")
            row_diag = {}

            err_keys = ["nuisance_error", "full_visual_error", "future_error",
                        "value_error", "action_error"]
            err_stats = {}
            for ek in err_keys:
                if ek in data:
                    err_stats[ek] = _stats(data[ek][mask])
            row_diag["error_stats"] = err_stats

            means = {k: v["mean"] for k, v in err_stats.items()}
            print(f"  {iname:25s} " +
                  "  ".join(f"{k.split('_')[0][:4]}={v:.4f}" for k, v in means.items()))

            # Per-sample pass rate
            checker = _CHECKERS.get(int(iid_val))
            if checker:
                idxs = np.where(mask)[0]
                passes = sum(1 for i in idxs if checker(_sample_at(data, i)))
                pass_rate = passes / len(idxs)
                row_diag["pass_rate"] = pass_rate
                ok = pass_rate >= MIN_PASS_RATE
                flag = "OK" if ok else "FAIL"
                print(f"    pass_rate={pass_rate:.3f} (≥{MIN_PASS_RATE}) → {flag}")
                if not ok:
                    all_ok = False

            diag["error_stats"][iname] = row_diag

        # weak_action_change rate
        if "weak_action_change" in data:
            ac_mask = iids == 3  # action_change id
            if ac_mask.sum() > 0:
                weak_rate = float(data["weak_action_change"][ac_mask].mean())
                diag["weak_action_change_rate"] = weak_rate
                if weak_rate <= WEAK_ACTION_WARN:
                    flag = "OK"
                elif weak_rate <= WEAK_ACTION_FAIL:
                    flag = "WARN"
                else:
                    flag = "FAIL"
                    all_ok = False
                print(f"  weak_action_change_rate={weak_rate:.3f} → {flag}")
                if weak_rate > WEAK_ACTION_WARN:
                    print(f"    [WARN] weak rate {weak_rate:.2f} > {WEAK_ACTION_WARN} (hard fail at {WEAK_ACTION_FAIL})")

        # Action-change relevance metadata rates (optional fields; skip if absent)
        ac_mask = iids == 3
        meta_fields = ["action_cell_on_path", "action_cell_near_path",
                       "action_path_action_changed", "action_path_len_changed"]
        if ac_mask.sum() > 0 and all(f in data for f in meta_fields):
            meta_rates = {f: float(data[f][ac_mask].mean()) for f in meta_fields}
            diag["action_meta_rates"] = meta_rates
            print("  action_change relevance rates: " +
                  "  ".join(f"{f.replace('action_', '')}={v:.3f}" for f, v in meta_rates.items()))

        local_action_fields = ["action_changed_count", "action_changed_cell_count",
                               "action_error_local"]
        if ac_mask.sum() > 0 and all(f in data for f in local_action_fields):
            local_stats = {f: float(data[f][ac_mask].mean()) for f in local_action_fields}
            diag["action_local_signal_stats"] = local_stats
            print("  action_change local signal: " +
                  "  ".join(f"mean_{f}={v:.3f}" for f, v in local_stats.items()))

        diagnostics[split_name] = diag

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
