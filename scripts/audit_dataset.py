"""
Strict scientific-validity audit for RemapBench datasets.

This is a STRICTER gate than remapbench.validate: validate.py checks that arrays
are well-formed and broad per-sample thresholds pass; audit_dataset.py judges
whether the dataset is *scientifically representative enough to train a pilot on*.

Usage:
    python scripts/audit_dataset.py --data_dir data/smoke --out data/smoke/audit_report.json
    python scripts/audit_dataset.py --data_dir data/remapbench_v0 \
        --out data/remapbench_v0/audit_report.json
    # add --strict to exit nonzero when pilot_ready is False

The audit writes a JSON report and prints a blunt PILOT READY: YES/NO summary.
By default it always exits 0 (so smoke pipelines proceed); the same hard-fail
rules are enforced regardless of dataset size, so a small smoke set may legitimately
report pilot_ready=False due to sample-count noise — this is printed clearly.
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
SINGLE_IDS = [0, 1, 2, 3]
TARGET_NAMES = ["sensory_update", "value_remap", "map_remap", "action_remap"]
SCALAR_KEYS = ["nuisance_error", "full_visual_error", "future_error",
               "value_error", "action_error"]
FEATURE_KEYS = SCALAR_KEYS + ["abs_path_len_change"]

NEAR_ZERO = 0.015      # "near zero" leakage threshold
SIGNAL    = 0.02       # "meaningful signal" threshold

# Channel indices (mirror remapbench.env)
CH_WALL, CH_START, CH_GOAL = 0, 1, 2
CH_NUISANCE, CH_DISTRACTOR, CH_DOOR = 3, 4, 5
ONE_WAY = [6, 7, 8, 9]

ALL_SPLITS = ["train_single", "val_single", "test_single",
              "test_composed", "test_larger", "test_noisy"]
SINGLE_SPLITS = ["train_single", "val_single", "test_single",
                 "test_larger", "test_noisy"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_split(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _get(data, key, n):
    """Fetch an array, or zeros of length n if the (optional) field is absent."""
    if key in data:
        return data[key]
    return np.zeros(n, dtype=np.float64)


def _abs_path_change(data):
    pa = data["path_len_after"].astype(np.int64)
    pb = data["path_len_before"].astype(np.int64)
    valid = (pa >= 0) & (pb >= 0)
    out = np.where(valid, np.abs(pa - pb), 0)
    return out.astype(np.float64)


def _stats(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return {k: float("nan") for k in
                ["mean", "std", "min", "max", "median", "iqr"]}
    q25, q75 = np.percentile(arr, [25, 75])
    return {
        "mean": float(arr.mean()), "std": float(arr.std()),
        "min": float(arr.min()), "max": float(arr.max()),
        "median": float(np.median(arr)), "iqr": float(q75 - q25),
    }


def _imbalance(counts):
    """max |count - mean| / mean over classes."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.sum() == 0:
        return float("nan")
    mean = counts.mean()
    if mean == 0:
        return float("nan")
    return float(np.max(np.abs(counts - mean)) / mean)


def _build_features(data):
    """Stack the scalar-error feature matrix [N, 6] for separability checks."""
    n = len(data["intervention_id"])
    cols = [np.asarray(_get(data, k, n), dtype=np.float64) for k in SCALAR_KEYS]
    cols.append(_abs_path_change(data))
    return np.stack(cols, axis=1)  # [N, 6]


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------
def audit(data_dir):
    splits = {}
    for name in ALL_SPLITS:
        path = os.path.join(data_dir, f"{name}.npz")
        if os.path.exists(path):
            splits[name] = load_split(path)

    report = {"data_dir": data_dir, "splits_present": list(splits.keys())}
    hard_fails = []   # each entry → blocks pilot
    warnings = []     # informational

    # =======================================================================
    # 1. Split integrity
    # =======================================================================
    integrity = {}

    # 1a. layout_id overlap
    layout_sets = {nm: set(d["layout_id"].tolist()) for nm, d in splits.items()
                   if "layout_id" in d}
    overlaps = {}
    names = list(layout_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ov = layout_sets[names[i]] & layout_sets[names[j]]
            if ov:
                overlaps[f"{names[i]}|{names[j]}"] = len(ov)
    integrity["layout_id_overlaps"] = overlaps
    if overlaps:
        hard_fails.append(f"layout_id overlap across splits: {overlaps}")

    # 1b. duplicate sample_id across all splits
    all_sids = []
    for d in splits.values():
        if "sample_id" in d:
            all_sids.extend(d["sample_id"].tolist())
    n_dup = len(all_sids) - len(set(all_sids))
    integrity["duplicate_sample_ids"] = int(n_dup)
    if n_dup > 0:
        hard_fails.append(f"{n_dup} duplicate sample_id(s) across splits")

    # 1c. class balance in single splits
    class_balance = {}
    for nm in SINGLE_SPLITS:
        if nm not in splits:
            continue
        iids = splits[nm]["intervention_id"]
        counts = [int((iids == c).sum()) for c in SINGLE_IDS]
        imb = _imbalance(counts)
        class_balance[nm] = {"counts": counts, "imbalance": imb}
        if imb == imb and imb > 0.10:  # not NaN and > 10%
            hard_fails.append(f"{nm}: class imbalance {imb:.2f} > 0.10 (counts {counts})")
    integrity["class_balance_single"] = class_balance

    # 1d. composed pair balance
    if "test_composed" in splits and "intervention_pair_id" in splits["test_composed"]:
        pid = splits["test_composed"]["intervention_pair_id"]
        uniq = sorted(set(int(x) for x in pid.tolist() if int(x) >= 0))
        counts = [int((pid == u).sum()) for u in uniq]
        imb = _imbalance(counts) if counts else float("nan")
        integrity["composed_pair_balance"] = {"pair_ids": uniq, "counts": counts,
                                              "imbalance": imb}
        if imb == imb and imb > 0.10:
            warnings.append(f"test_composed: pair imbalance {imb:.2f} > 0.10")

    report["1_split_integrity"] = integrity

    # =======================================================================
    # NaN / Inf check (hard fail)
    # =======================================================================
    nan_problems = {}
    for nm, d in splits.items():
        bad = []
        for k in SCALAR_KEYS + ["delta_future", "delta_value", "action_delta",
                                "value_after", "future_after"]:
            if k in d and not np.isfinite(d[k].astype(np.float64)).all():
                bad.append(k)
        if bad:
            nan_problems[nm] = bad
    report["nan_inf_problems"] = nan_problems
    if nan_problems:
        hard_fails.append(f"NaN/Inf found: {nan_problems}")

    # =======================================================================
    # 2. Signal strength by class
    # =======================================================================
    signal = {}
    for nm in SINGLE_SPLITS:
        if nm not in splits:
            continue
        d = splits[nm]
        n = len(d["intervention_id"])
        iids = d["intervention_id"]
        apc = _abs_path_change(d)
        per_class = {}
        for c in SINGLE_IDS:
            mask = iids == c
            if mask.sum() == 0:
                continue
            cname = INTERVENTION_NAMES[c]
            entry = {}
            for k in SCALAR_KEYS:
                entry[k] = _stats(_get(d, k, n)[mask])
            entry["path_len_before"] = _stats(d["path_len_before"][mask])
            entry["path_len_after"] = _stats(d["path_len_after"][mask])
            entry["abs_path_len_change"] = _stats(apc[mask])
            per_class[cname] = entry
        signal[nm] = per_class
    report["2_signal_strength"] = signal

    # =======================================================================
    # 3. Signal separability (nearest-centroid, numpy only)
    # =======================================================================
    sep = {}
    if "train_single" in splits:
        tr = splits["train_single"]
        Xtr = _build_features(tr)
        ytr = tr["intervention_id"]
        mu = Xtr.mean(0)
        sd = Xtr.std(0) + 1e-8
        Xtr_z = (Xtr - mu) / sd

        centroids = {}
        for c in SINGLE_IDS:
            mask = ytr == c
            if mask.sum() > 0:
                centroids[c] = Xtr_z[mask].mean(0)
        cids = sorted(centroids.keys())
        C = np.stack([centroids[c] for c in cids], axis=0)  # [n_classes, 6]

        def _nc_accuracy(X, y):
            Xz = (X - mu) / sd
            # distances to each centroid
            d2 = ((Xz[:, None, :] - C[None, :, :]) ** 2).sum(-1)  # [N, n_classes]
            pred = np.array(cids)[d2.argmin(1)]
            return float((pred == y).mean())

        sep["train_self_accuracy"] = _nc_accuracy(Xtr, ytr)
        for tnm in ["test_single", "test_larger", "test_noisy"]:
            if tnm in splits:
                d = splits[tnm]
                # only single-class samples
                mask = np.isin(d["intervention_id"], SINGLE_IDS)
                if mask.sum() > 0:
                    sep[f"{tnm}_nearest_centroid_accuracy"] = _nc_accuracy(
                        _build_features(d)[mask], d["intervention_id"][mask])

        # Top distinguishing scalar feature per class (largest |z-centroid|)
        distinguishing = {}
        for c in cids:
            z = centroids[c]
            order = np.argsort(-np.abs(z))
            distinguishing[INTERVENTION_NAMES[c]] = [
                {"feature": FEATURE_KEYS[idx], "centroid_z": float(z[idx])}
                for idx in order[:3]
            ]
        sep["top_distinguishing_features"] = distinguishing

        acc = sep.get("test_single_nearest_centroid_accuracy", float("nan"))
        if acc == acc:
            if acc < 0.5:
                warnings.append(
                    f"scalar separability low (test_single NC acc={acc:.2f}); "
                    "labels may be noisy or signal weak")
            elif acc >= 0.999:
                warnings.append(
                    f"scalar separability perfect (NC acc={acc:.2f}); "
                    "labels may be trivially recoverable from scalar artifacts")
    report["3_separability"] = sep

    # =======================================================================
    # 4. Behavioral meaningfulness
    # =======================================================================
    behavior = {}
    for nm in SINGLE_SPLITS:
        if nm not in splits:
            continue
        d = splits[nm]
        n = len(d["intervention_id"])
        iids = d["intervention_id"]
        apc = _abs_path_change(d)
        fe = _get(d, "future_error", n)
        ve = _get(d, "value_error", n)
        b = {}

        # topology
        tmask = iids == 2
        if tmask.sum() > 0:
            b["topology"] = {
                "n": int(tmask.sum()),
                "frac_path_changed": float((apc[tmask] >= 1).mean()),
                "frac_future_signal": float((fe[tmask] >= SIGNAL).mean()),
                "frac_value_signal": float((ve[tmask] >= SIGNAL).mean()),
            }
            meaningful = (apc[tmask] >= 1) | (fe[tmask] >= SIGNAL) | (ve[tmask] >= SIGNAL)
            b["topology"]["frac_meaningful"] = float(meaningful.mean())

        # action
        amask = iids == 3
        if amask.sum() > 0:
            weak = _get(d, "weak_action_change", n)[amask]
            on_path = _get(d, "action_cell_on_path", n)[amask]
            b["action"] = {
                "n": int(amask.sum()),
                "weak_rate": float(weak.mean()),
                "frac_path_changed": float((apc[amask] >= 1).mean()),
                "frac_future_signal": float((fe[amask] >= SIGNAL).mean()),
                "frac_value_signal": float((ve[amask] >= SIGNAL).mean()),
                "frac_on_path": float(on_path.mean()),
            }
            meaningful = ((apc[amask] >= 1) | (fe[amask] >= SIGNAL) |
                          (ve[amask] >= SIGNAL) | (on_path >= 0.5))
            b["action"]["frac_meaningful"] = float(meaningful.mean())
        behavior[nm] = b

    report["4_behavioral_meaningfulness"] = behavior

    # Hard-fail rules on train/test single splits
    for nm in ["train_single", "test_single"]:
        if nm not in behavior:
            continue
        bt = behavior[nm].get("topology")
        if bt and bt["frac_meaningful"] < 0.70:
            hard_fails.append(f"{nm}: topology meaningful fraction "
                              f"{bt['frac_meaningful']:.2f} < 0.70")
        ba = behavior[nm].get("action")
        if ba:
            if ba["weak_rate"] > 0.20:
                hard_fails.append(f"{nm}: weak_action_change_rate "
                                  f"{ba['weak_rate']:.2f} > 0.20")
            if ba["frac_meaningful"] < 0.50:
                hard_fails.append(f"{nm}: action meaningful fraction "
                                  f"{ba['frac_meaningful']:.2f} < 0.50")

    # =======================================================================
    # 5. Artifact checks (channel leakage on single classes)
    # =======================================================================
    # Forbidden channels that must NOT change per intervention type.
    forbidden = {
        0: [CH_WALL, CH_START, CH_GOAL] + ONE_WAY,            # sensory_nuisance
        1: [CH_WALL, CH_START] + ONE_WAY,                     # goal_relocation
        2: [CH_START, CH_GOAL] + ONE_WAY,                     # topology_change
        3: [CH_WALL, CH_START, CH_GOAL],                      # action_change
    }
    artifacts = {}
    for nm in SINGLE_SPLITS:
        if nm not in splits:
            continue
        d = splits[nm]
        if "before_grid" not in d or "after_grid" not in d:
            continue
        iids = d["intervention_id"]
        before = d["before_grid"].astype(np.int16)
        after = d["after_grid"].astype(np.int16)
        diff = np.abs(after - before)  # [N, C, H, W]
        viol = {}
        for c, chans in forbidden.items():
            mask = iids == c
            if mask.sum() == 0:
                continue
            changed = (diff[mask][:, chans].reshape(mask.sum(), -1).sum(1) > 0)
            n_viol = int(changed.sum())
            if n_viol > 0:
                viol[INTERVENTION_NAMES[c]] = {
                    "n_violations": n_viol, "n_class": int(mask.sum()),
                    "forbidden_channels": chans,
                }
        artifacts[nm] = viol

    report["5_artifact_checks"] = artifacts
    # Channel-leakage violations are hard fails (they break label semantics)
    for nm, viol in artifacts.items():
        for cname, info in viol.items():
            hard_fails.append(f"{nm}: {cname} changed forbidden channels "
                              f"{info['forbidden_channels']} in {info['n_violations']} samples")

    # full_visual_error distribution by class (informational artifact view)
    fve_bins = {}
    for nm in SINGLE_SPLITS:
        if nm not in splits:
            continue
        d = splits[nm]
        n = len(d["intervention_id"])
        fve = _get(d, "full_visual_error", n)
        iids = d["intervention_id"]
        per = {}
        for c in SINGLE_IDS:
            mask = iids == c
            if mask.sum() > 0:
                per[INTERVENTION_NAMES[c]] = _stats(fve[mask])
        fve_bins[nm] = per
    report["5_full_visual_error_by_class"] = fve_bins

    # =======================================================================
    # Scalar leakage hard-fail rules (sensory / goal)
    # =======================================================================
    leakage = {}
    for nm in ["train_single", "test_single"]:
        if nm not in splits:
            continue
        d = splits[nm]
        n = len(d["intervention_id"])
        iids = d["intervention_id"]
        fe = _get(d, "future_error", n)
        ve = _get(d, "value_error", n)
        ae = _get(d, "action_error", n)
        ent = {}

        smask = iids == 0  # sensory_nuisance: no structural signal allowed
        if smask.sum() > 0:
            leak = (fe[smask] > NEAR_ZERO) | (ve[smask] > NEAR_ZERO) | (ae[smask] > NEAR_ZERO)
            frac = float(leak.mean())
            ent["sensory_structural_leak_frac"] = frac
            if frac > 0.05:
                hard_fails.append(f"{nm}: sensory_nuisance structural leak "
                                  f"{frac:.2f} > 0.05")

        gmask = iids == 1  # goal_relocation: no action (transition) signal allowed
        if gmask.sum() > 0:
            leak = ae[gmask] > NEAR_ZERO
            frac = float(leak.mean())
            ent["goal_action_leak_frac"] = frac
            if frac > 0.05:
                hard_fails.append(f"{nm}: goal_relocation action leak "
                                  f"{frac:.2f} > 0.05")
        leakage[nm] = ent
    report["5b_scalar_leakage"] = leakage

    # =======================================================================
    # Composed validity
    # =======================================================================
    if "test_composed" in splits:
        mh = splits["test_composed"]["target_multihot"]
        sums = mh.sum(axis=1)
        n_bad = int((sums < 2).sum())
        report["composed_target_sum_invalid"] = n_bad
        if n_bad > 0:
            hard_fails.append(f"test_composed: {n_bad} samples with <2 active labels")

    # =======================================================================
    # 6. Pilot readiness decision
    # =======================================================================
    pilot_ready = len(hard_fails) == 0
    report["hard_fails"] = hard_fails
    report["warnings"] = warnings
    report["pilot_ready"] = pilot_ready

    # Small-dataset note
    small = any(nm in splits and len(splits[nm]["intervention_id"]) < 400
                for nm in ["train_single"])
    report["small_dataset_note"] = small

    return report


def _print_summary(report):
    print("\n" + "=" * 64)
    print("DATASET AUDIT SUMMARY")
    print("=" * 64)
    print(f"splits present: {report['splits_present']}")

    sep = report.get("3_separability", {})
    if "train_self_accuracy" in sep:
        print(f"\nnearest-centroid separability (scalar errors only):")
        print(f"  train self      = {sep['train_self_accuracy']:.3f}")
        for k, v in sep.items():
            if k.endswith("_nearest_centroid_accuracy"):
                print(f"  {k.replace('_nearest_centroid_accuracy',''):15s} = {v:.3f}")

    beh = report.get("4_behavioral_meaningfulness", {})
    for nm in ["train_single", "test_single"]:
        if nm in beh:
            bt = beh[nm].get("topology", {})
            ba = beh[nm].get("action", {})
            if bt:
                print(f"\n[{nm}] topology meaningful frac = {bt.get('frac_meaningful', float('nan')):.3f}")
            if ba:
                print(f"[{nm}] action   meaningful frac = {ba.get('frac_meaningful', float('nan')):.3f} "
                      f"| weak_rate = {ba.get('weak_rate', float('nan')):.3f} "
                      f"| on_path = {ba.get('frac_on_path', float('nan')):.3f}")

    if report["warnings"]:
        print("\nWARNINGS (non-blocking):")
        for w in report["warnings"]:
            print(f"  - {w}")

    if report["hard_fails"]:
        print("\nHARD FAILS (block pilot):")
        for f in report["hard_fails"]:
            print(f"  - {f}")

    if report.get("small_dataset_note"):
        print("\nNOTE: train_single < 400 samples — this looks like a smoke/preflight set. "
              "pilot_ready may be False purely due to small-sample noise; the same rules "
              "are enforced on full data.")

    print("\n" + "-" * 64)
    print(f"PILOT READY: {'YES' if report['pilot_ready'] else 'NO'}")
    print("-" * 64)


def main():
    parser = argparse.ArgumentParser(description="Strict RemapBench dataset audit")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default: <data_dir>/audit_report.json)")
    parser.add_argument("--strict", action="store_true",
                        help="Exit nonzero if pilot_ready is False")
    args = parser.parse_args()

    report = audit(args.data_dir)
    _print_summary(report)

    out_path = args.out or os.path.join(args.data_dir, "audit_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nAudit report written to {out_path}")

    if args.strict and not report["pilot_ready"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
