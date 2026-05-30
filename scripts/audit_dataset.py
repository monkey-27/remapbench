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

# Default COMPOSED_PAIRS ordering (mirrors remapbench.interventions.COMPOSED_PAIRS).
# Used to map intervention_pair_id → component intervention names when metadata.json
# is unavailable.
DEFAULT_COMPOSED_PAIRS = [
    ("goal_relocation",  "topology_change"),   # 0
    ("sensory_nuisance", "action_change"),     # 1
    ("goal_relocation",  "action_change"),     # 2
    ("sensory_nuisance", "topology_change"),   # 3
]

MIN_SAVED_FRACTION = 0.95   # split-completion hard-fail threshold
COMPOSED_EVID_MIN  = 0.85   # composed component evidence pass-rate threshold
ACTION_ERR_MIN     = 0.012  # legacy global action_error reporting threshold
ACTION_LOCAL_MIN   = 0.20   # local transition-change density threshold


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

    # Load metadata.json if present (for requested split sizes + composed pair map)
    metadata = None
    meta_path = os.path.join(data_dir, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            metadata = None
            warnings.append("metadata.json present but could not be parsed")

    # Resolve composed pair map (id -> (compA, compB))
    composed_pairs = list(DEFAULT_COMPOSED_PAIRS)
    if metadata and isinstance(metadata.get("composed_pairs"), list) and metadata["composed_pairs"]:
        try:
            composed_pairs = [tuple(p) for p in metadata["composed_pairs"]]
        except TypeError:
            pass

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
    # 1b. Split completion (requested vs actual saved counts)
    # =======================================================================
    completion = {}
    if metadata and isinstance(metadata.get("requested_split_sizes"), dict):
        req = metadata["requested_split_sizes"]
        # Was this generated as a tiny smoke set? (train_single requested < 400)
        smoke_sized = int(req.get("train_single", 0)) < 400
        for split_name, requested_n in req.items():
            requested_n = int(requested_n)
            if requested_n <= 0:
                continue
            actual_n = (len(splits[split_name]["intervention_id"])
                        if split_name in splits else 0)
            frac = actual_n / requested_n
            completion[split_name] = {
                "requested": requested_n,
                "actual": int(actual_n),
                "saved_fraction": float(frac),
            }
            if actual_n == 0:
                hard_fails.append(
                    f"{split_name}: requested {requested_n} samples but 0 saved "
                    "(generation produced nothing)")
            elif frac < MIN_SAVED_FRACTION:
                hard_fails.append(
                    f"{split_name}: only {actual_n}/{requested_n} saved "
                    f"(fraction {frac:.2f} < {MIN_SAVED_FRACTION})"
                    + (" [smoke-sized: see note]" if smoke_sized else ""))
        completion["_smoke_sized"] = bool(smoke_sized)
    else:
        warnings.append("metadata.json missing or has no requested_split_sizes; "
                        "cannot verify requested split completion")
    report["1b_split_completion"] = completion

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
            changed_count = _get(d, "action_changed_count", n)[amask]
            changed_cell_count = _get(d, "action_changed_cell_count", n)[amask]
            action_local = _get(d, "action_error_local", n)[amask]
            b["action"] = {
                "n": int(amask.sum()),
                "weak_rate": float(weak.mean()),
                "frac_path_changed": float((apc[amask] >= 1).mean()),
                "frac_future_signal": float((fe[amask] >= SIGNAL).mean()),
                "frac_value_signal": float((ve[amask] >= SIGNAL).mean()),
                "frac_on_path": float(on_path.mean()),
                "mean_action_changed_count": float(changed_count.mean()),
                "mean_action_changed_cell_count": float(changed_cell_count.mean()),
                "mean_action_error_local": float(action_local.mean()),
                "frac_action_changed_count_zero": float((changed_count < 1).mean()),
                "frac_action_changed_cell_count_zero": float((changed_cell_count < 1).mean()),
                "frac_action_error_local_below_min": float((action_local < ACTION_LOCAL_MIN).mean()),
            }
            meaningful = (
                (changed_count >= 1) & (
                    (weak == 0) | (apc[amask] >= 1) | (fe[amask] >= SIGNAL) |
                    (ve[amask] >= SIGNAL) | (on_path >= 0.5)
                )
            )
            b["action"]["frac_meaningful"] = float(meaningful.mean())
        behavior[nm] = b

    report["4_behavioral_meaningfulness"] = behavior

    # Hard-fail rules on single splits. Topology is gated on train/test single,
    # while action evidence is checked anywhere action_change appears.
    for nm in SINGLE_SPLITS:
        if nm not in behavior:
            continue
        bt = behavior[nm].get("topology")
        if nm in ["train_single", "test_single"] and bt and bt["frac_meaningful"] < 0.70:
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
            if ba["frac_action_changed_count_zero"] > 0.02:
                hard_fails.append(
                    f"{nm}: action_changed_count < 1 in "
                    f"{ba['frac_action_changed_count_zero']:.2f} of action samples (>0.02)")
            if ba["frac_action_changed_cell_count_zero"] > 0.02:
                hard_fails.append(
                    f"{nm}: action_changed_cell_count < 1 in "
                    f"{ba['frac_action_changed_cell_count_zero']:.2f} of action samples (>0.02)")
            if ba["frac_action_error_local_below_min"] > 0.02:
                hard_fails.append(
                    f"{nm}: action_error_local < {ACTION_LOCAL_MIN} in "
                    f"{ba['frac_action_error_local_below_min']:.2f} of action samples (>0.02)")

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
    # 6. Composed intervention evidence audit
    # =======================================================================
    # For each composed sample, verify scalar evidence supports each active
    # component intervention (not just that target_multihot has >=2 labels).
    composed_evidence = {}
    if "test_composed" in splits:
        d = splits["test_composed"]
        n = len(d["intervention_id"])
        ne = _get(d, "nuisance_error", n)
        fe = _get(d, "future_error", n)
        ve = _get(d, "value_error", n)
        ae = _get(d, "action_error", n)
        apc = _abs_path_change(d)
        weak = _get(d, "weak_action_change", n)
        on_path = _get(d, "action_cell_on_path", n)
        changed_count = _get(d, "action_changed_count", n)
        action_local = _get(d, "action_error_local", n)

        def _component_evidence(comp):
            """Boolean array [n]: does scalar evidence support `comp`?"""
            if comp == "sensory_nuisance":
                return ne >= SIGNAL
            if comp == "goal_relocation":
                return ve >= SIGNAL
            if comp == "topology_change":
                return (fe >= SIGNAL) | (apc >= 1) | (ve >= SIGNAL)
            if comp == "action_change":
                return (changed_count >= 1) & (action_local >= ACTION_LOCAL_MIN) & (
                    (weak == 0) | (apc >= 1) | (fe >= SIGNAL) |
                    (ve >= SIGNAL) | (on_path >= 0.5))
            return np.zeros(n, dtype=bool)

        pid = d["intervention_pair_id"] if "intervention_pair_id" in d else None
        if pid is not None:
            uniq = sorted(set(int(x) for x in pid.tolist() if int(x) >= 0))
        else:
            uniq = []

        if not uniq:
            # No pair ids — cannot attribute evidence to components.
            warnings.append("test_composed: intervention_pair_id absent; "
                            "cannot run composed component-evidence audit")
        for u in uniq:
            if u >= len(composed_pairs):
                warnings.append(f"test_composed: pair_id {u} out of range of "
                                f"known composed_pairs ({len(composed_pairs)})")
                continue
            compA, compB = composed_pairs[u][0], composed_pairs[u][1]
            mask = (pid == u)
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            evidA = _component_evidence(compA)[mask]
            evidB = _component_evidence(compB)[mask]
            both = evidA & evidB
            entry = {
                "pair": [compA, compB],
                "n": cnt,
                "component_pass_rates": {
                    compA: float(evidA.mean()),
                    compB: float(evidB.mean()),
                },
                "all_components_pass_rate": float(both.mean()),
                "mean_scalar_errors": {
                    "nuisance_error": float(ne[mask].mean()),
                    "future_error": float(fe[mask].mean()),
                    "value_error": float(ve[mask].mean()),
                    "action_error": float(ae[mask].mean()),
                    "action_changed_count": float(changed_count[mask].mean()),
                    "action_error_local": float(action_local[mask].mean()),
                    "abs_path_len_change": float(apc[mask].mean()),
                },
            }
            if "action_change" in (compA, compB):
                entry["weak_action_change_rate"] = float(weak[mask].mean())
            composed_evidence[f"pair_{u}"] = entry

            # Hard-fail rules
            if entry["all_components_pass_rate"] < COMPOSED_EVID_MIN:
                hard_fails.append(
                    f"test_composed[{compA}+{compB}]: all-components evidence rate "
                    f"{entry['all_components_pass_rate']:.2f} < {COMPOSED_EVID_MIN}")
            for comp, rate in entry["component_pass_rates"].items():
                if rate < COMPOSED_EVID_MIN:
                    hard_fails.append(
                        f"test_composed[{compA}+{compB}]: component '{comp}' evidence "
                        f"rate {rate:.2f} < {COMPOSED_EVID_MIN}")
    report["6_composed_evidence"] = composed_evidence

    # =======================================================================
    # 7. Action metadata consistency
    # =======================================================================
    action_meta = {}

    # 7a. Single splits
    for nm in SINGLE_SPLITS:
        if nm not in splits:
            continue
        d = splits[nm]
        n = len(d["intervention_id"])
        iids = d["intervention_id"]
        if "action_cell_row" not in d:
            continue
        row = d["action_cell_row"].astype(np.int64)
        col = d["action_cell_col"].astype(np.int64)
        ae = _get(d, "action_error", n)
        changed_count = _get(d, "action_changed_count", n)
        changed_cell_count = _get(d, "action_changed_cell_count", n)
        action_local = _get(d, "action_error_local", n)
        weak = _get(d, "weak_action_change", n)
        on_path = _get(d, "action_cell_on_path", n)
        near_path = _get(d, "action_cell_near_path", n)
        apc_present = _get(d, "action_path_action_changed", n)
        plc = _get(d, "action_path_len_changed", n)
        ent = {}

        amask = iids == 3  # action_change
        if amask.sum() > 0:
            na = int(amask.sum())
            missing_cell = ((row[amask] < 0) | (col[amask] < 0))
            frac_missing = float(missing_cell.mean())
            # at least one of on/near path should be 1 unless weak
            wk = weak[amask]
            located = (on_path[amask] >= 0.5) | (near_path[amask] >= 0.5)
            frac_unlocated_strong = float(((wk == 0) & ~located).mean())
            frac_action_err_low = float((ae[amask] <= ACTION_ERR_MIN).mean())
            frac_changed_count_zero = float((changed_count[amask] < 1).mean())
            frac_changed_cell_count_zero = float((changed_cell_count[amask] < 1).mean())
            frac_action_local_low = float((action_local[amask] < ACTION_LOCAL_MIN).mean())
            ent["action_change"] = {
                "n": na,
                "frac_missing_cell": frac_missing,
                "frac_strong_but_unlocated": frac_unlocated_strong,
                "frac_action_error_below_min": frac_action_err_low,
                "frac_action_changed_count_zero": frac_changed_count_zero,
                "frac_action_changed_cell_count_zero": frac_changed_cell_count_zero,
                "frac_action_error_local_below_min": frac_action_local_low,
            }
            if frac_missing > 0.02:
                hard_fails.append(
                    f"{nm}: {frac_missing:.2f} of action_change samples missing "
                    "action cell metadata (>0.02)")
            if frac_changed_count_zero > 0.02:
                hard_fails.append(
                    f"{nm}: {frac_changed_count_zero:.2f} of action_change samples have "
                    "action_changed_count < 1 (>0.02)")
            if frac_changed_cell_count_zero > 0.02:
                hard_fails.append(
                    f"{nm}: {frac_changed_cell_count_zero:.2f} of action_change samples have "
                    "action_changed_cell_count < 1 (>0.02)")
            if frac_action_local_low > 0.02:
                hard_fails.append(
                    f"{nm}: {frac_action_local_low:.2f} of action_change samples have "
                    f"action_error_local < {ACTION_LOCAL_MIN} (>0.02)")

        # non-action single classes must have default (cleared) metadata
        nmask = np.isin(iids, [0, 1, 2])
        if nmask.sum() > 0:
            nn = int(nmask.sum())
            cell_set = ((row[nmask] >= 0) | (col[nmask] >= 0))
            bools_set = ((on_path[nmask] >= 0.5) | (near_path[nmask] >= 0.5) |
                         (apc_present[nmask] >= 0.5) | (plc[nmask] >= 0.5))
            frac_leak = float((cell_set | bools_set).mean())
            ent["non_action"] = {
                "n": nn,
                "frac_metadata_set": frac_leak,
            }
            if frac_leak > 0.0:
                hard_fails.append(
                    f"{nm}: {frac_leak:.3f} of non-action single samples have "
                    "action metadata set (>0)")
        action_meta[nm] = ent

    # 7b. Composed splits with an action component
    if "test_composed" in splits and "action_cell_row" in splits["test_composed"]:
        d = splits["test_composed"]
        n = len(d["intervention_id"])
        row = d["action_cell_row"].astype(np.int64)
        col = d["action_cell_col"].astype(np.int64)
        changed_count = _get(d, "action_changed_count", n)
        action_local = _get(d, "action_error_local", n)
        pid = d["intervention_pair_id"] if "intervention_pair_id" in d else None
        if pid is not None:
            action_pair_ids = [u for u in range(len(composed_pairs))
                               if "action_change" in composed_pairs[u]]
            amask = np.isin(pid, action_pair_ids)
            if amask.sum() > 0:
                na = int(amask.sum())
                missing_cell = ((row[amask] < 0) | (col[amask] < 0))
                frac_missing = float(missing_cell.mean())
                frac_changed_ok = float((changed_count[amask] >= 1).mean())
                frac_action_local_ok = float((action_local[amask] >= ACTION_LOCAL_MIN).mean())
                action_meta["test_composed"] = {
                    "n_action_composed": na,
                    "frac_missing_cell": frac_missing,
                    "frac_action_changed_count_above_min": frac_changed_ok,
                    "frac_action_error_local_above_min": frac_action_local_ok,
                }
                if frac_missing > 0.15:
                    hard_fails.append(
                        f"test_composed: {frac_missing:.2f} of action-composed "
                        "samples missing action cell metadata (>0.15)")
                if frac_changed_ok < COMPOSED_EVID_MIN:
                    hard_fails.append(
                        f"test_composed: only {frac_changed_ok:.2f} of "
                        "action-composed samples have action_changed_count >= 1")
                if frac_action_local_ok < COMPOSED_EVID_MIN:
                    hard_fails.append(
                        f"test_composed: only {frac_action_local_ok:.2f} of "
                        f"action-composed samples have action_error_local >= {ACTION_LOCAL_MIN}")
    report["7_action_metadata_consistency"] = action_meta

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

    comp = report.get("1b_split_completion", {})
    if comp:
        print("\nsplit completion (requested -> actual saved):")
        for k, v in comp.items():
            if k.startswith("_"):
                continue
            print(f"  {k:15s} {v['actual']:>6d}/{v['requested']:<6d} "
                  f"(fraction {v['saved_fraction']:.3f})")

    ce = report.get("6_composed_evidence", {})
    if ce:
        print("\ncomposed component evidence (all-components pass rate):")
        for k, v in ce.items():
            pair = "+".join(v["pair"])
            print(f"  {pair:38s} all={v['all_components_pass_rate']:.3f} "
                  f"(n={v['n']})")

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

    # Split-completion smoke caveat
    comp = report.get("1b_split_completion", {})
    completion_fail = any(
        ("only" in f or "0 saved" in f) and "saved" in f
        for f in report["hard_fails"])
    if comp.get("_smoke_sized") and completion_fail:
        print("\nNOTE (split completion): one or more splits saved < 95% of requested "
              "samples. This may be due to tiny smoke size, but full pilot must pass "
              "this rule.")

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
