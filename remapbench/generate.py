"""
Dataset generation CLI.
Usage:
    python -m remapbench.generate --out data/remapbench_v0 --seed 0 \
        --grid_size 10 --n_train 12000 --n_val 2000 --n_test 2000 --n_composed 2000

Each sample gets a globally unique layout_id derived from the split's seed offset
plus its position index, guaranteeing disjoint layout spaces across splits.
"""
import argparse
import json
import os
import time
import numpy as np

from .env import make_grid, N_CHANNELS
from .oracle import build_sample_arrays
from .interventions import (
    apply_intervention, INTERVENTION_IDS, COMPOSED_PAIRS,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG = {
    "grid_size": 10,
    "gamma": 0.95,
    "wall_prob": 0.25,
    "min_path": 6,
    "max_sample_tries": 50,
    "max_layout_tries": 200,
}

# Post-hoc oracle thresholds — reject samples with insufficient signal
ORACLE_THRESHOLDS = {
    "sensory_nuisance": {"nuisance_error": 0.02},
    "goal_relocation":  {"value_error": 0.02},
    "topology_change":  {"future_error": 0.012},   # > validate THRESH_HIGH=0.01
}

ACTION_LOCAL_MIN = 0.20

SINGLE_INTERVENTIONS = ["sensory_nuisance", "goal_relocation", "topology_change", "action_change"]

# ---------------------------------------------------------------------------
# Collision-proof split identity scheme
# ---------------------------------------------------------------------------
# Each split has a unique integer id. Layout IDs and RNG seeds are derived
# arithmetically from (seed, split_id, sample_index, attempt) so that:
#   - layout_id is PROVABLY disjoint across splits (depends on split_id and index
#     only, NOT on retry attempt).
#   - layout_seed / intervention_seed are disjoint across splits and across the
#     layout/intervention roles within a split.
# This replaces the old additive SEED_OFFSETS scheme, which could in principle
# collide across splits.
SPLIT_IDS = {
    "train_single":  0,
    "val_single":    1,
    "test_single":   2,
    "test_composed": 3,
    "test_larger":   4,
    "test_noisy":    5,
    "train_composed_seen":   6,
    "val_composed_seen":     7,
    "test_composed_seen":    8,
    "test_composed_heldout": 9,
}

LAYOUT_ID_STRIDE = 1_000_000_000   # layout_id = split_id * LAYOUT_ID_STRIDE + index
SPLIT_SEED_STRIDE = 100_000_000    # per-split seed block
ROLE_OFFSET = 50_000_000           # layout seeds in [0, ROLE), intervention in [ROLE, 2*ROLE)
BASE_SEED_STRIDE = 10_000_000_000  # per global-seed block

# DEPRECATED: retained only for backward reference in metadata; no longer used
# for layout uniqueness (see SPLIT_IDS scheme above).
SEED_OFFSETS = {
    "train_single":   0,
    "val_single":     500_000,
    "test_single":    1_000_000,
    "test_composed":  1_500_000,
    "test_larger":    2_000_000,
    "test_noisy":     2_500_000,
}


# ---------------------------------------------------------------------------
# Core sample generation
# ---------------------------------------------------------------------------

def _generate_one(
    layout_seed, intervention_seed, intervention_type, H, W, gamma, cfg,
    composed_pair_idx=None, sample_id=0, layout_id=0,
    nuisance_prob=0.15, distractor_prob=0.10,
):
    """Generate one sample with fully deterministic, per-sample seeds.

    layout_id is supplied explicitly (collision-proof per-split scheme) and is
    independent of the retry attempt encoded in layout_seed.
    """
    rng_layout = np.random.default_rng(layout_seed)
    rng_inter  = np.random.default_rng(intervention_seed)

    grid, start, goal, _ = make_grid(
        H, W, rng_layout,
        wall_prob=cfg["wall_prob"],
        min_path=cfg["min_path"],
        nuisance_prob=nuisance_prob,
        distractor_prob=distractor_prob,
    )

    result = apply_intervention(grid, start, goal, rng_inter, intervention_type, composed_pair_idx)
    g2, g2_goal, iname, multihot, weak, extra_meta = result
    pair = extra_meta.get("composed_pair")
    pair_id = COMPOSED_PAIRS.index(pair) if pair in COMPOSED_PAIRS else -1

    iid = INTERVENTION_IDS.get(iname, INTERVENTION_IDS["composed"])
    sample = build_sample_arrays(
        grid, g2, start, goal, g2_goal,
        iid, multihot, gamma,
        layout_id=layout_id,
        sample_id=sample_id,
        layout_seed=layout_seed,
        intervention_seed=intervention_seed,
        intervention_pair_id=pair_id,
        weak_action_change=weak,
        action_meta=extra_meta,
    )

    # Post-hoc oracle threshold check (minimum signal)
    thresholds = ORACLE_THRESHOLDS.get(intervention_type, {})
    for err_key, min_val in thresholds.items():
        if float(sample[err_key]) < min_val:
            raise ValueError(
                f"{intervention_type}: {err_key}={float(sample[err_key]):.4f} < {min_val}"
            )

    # Representativeness / behavioral-meaningfulness checks (PART 4 & 5)
    pa, pb = int(sample["path_len_after"]), int(sample["path_len_before"])
    abs_path_change = abs(pa - pb) if (pa >= 0 and pb >= 0) else 0
    fe = float(sample["future_error"])
    ve = float(sample["value_error"])

    def _action_component_ok():
        changed_count = int(sample["action_changed_count"])
        changed_cell_count = int(sample["action_changed_cell_count"])
        action_local = float(sample["action_error_local"])
        on_path = int(sample["action_cell_on_path"])
        meaningful = (
            int(weak) == 0 or abs_path_change >= 1 or ve >= 0.02 or
            fe >= 0.02 or on_path == 1
        )
        return (
            changed_count >= 1 and
            changed_cell_count >= 1 and
            action_local >= ACTION_LOCAL_MIN and
            meaningful
        )

    if intervention_type == "action_change":
        # Reject weak/off-path action samples with no transition-local evidence.
        meaningful = _action_component_ok()
        if not meaningful:
            raise ValueError(
                "action_change: insufficient scale-invariant transition evidence")

    if intervention_type == "topology_change":
        # Reject cosmetic wall perturbations with no structural effect.
        meaningful = (fe >= 0.02 or abs_path_change >= 1 or ve >= 0.02)
        if not meaningful:
            raise ValueError("topology_change: not representative (no future/path/value effect)")

    if intervention_type == "composed":
        # Each ACTIVE component of a composed sample must be supported by scalar
        # evidence at the COMBINED-sample level — not merely present in the
        # multihot label. These rules mirror audit_dataset.py's composed-evidence
        # audit exactly, so a dataset that generates cleanly also passes the audit.
        ne = float(sample["nuisance_error"])
        mh = list(multihot)
        # mh = [sensory_update, value_remap, map_remap, action_remap]
        if mh[0] and not (ne >= 0.02):
            raise ValueError("composed: sensory component lacks nuisance signal")
        if mh[1] and not (ve >= 0.02):
            raise ValueError("composed: value component lacks value signal")
        if mh[2] and not (fe >= 0.02 or abs_path_change >= 1 or ve >= 0.02):
            raise ValueError("composed: topology component lacks structural effect")
        if mh[3]:
            action_ok = _action_component_ok()
            if not action_ok:
                raise ValueError("composed: action component lacks transition effect")

    return sample


def generate_split(
    n, intervention_schedule, split_name, split_id, base_seed, H, W, gamma, cfg, verbose=True,
    composed_pair_schedule=None, global_id_offset=0,
    nuisance_prob=0.15, distractor_prob=0.10,
):
    """Generate n samples for a split using the collision-proof seed scheme.

    layout_id      = split_id * LAYOUT_ID_STRIDE + i        (attempt-independent)
    layout_seed    = base_seed + split_id*SPLIT_SEED_STRIDE + i*stride + attempt
    inter_seed     = layout_seed + ROLE_OFFSET
    where stride = cfg["max_sample_tries"] (so attempts never overlap across i).
    """
    samples = []
    failures = 0
    t0 = time.time()
    stride = cfg["max_sample_tries"]
    split_base = base_seed + split_id * SPLIT_SEED_STRIDE

    for i, itype in enumerate(intervention_schedule):
        cidx = (composed_pair_schedule[i] if composed_pair_schedule else None)
        global_sample_id = global_id_offset + i
        layout_id = split_id * LAYOUT_ID_STRIDE + i  # disjoint across splits, fixed per index
        success = False

        for attempt in range(stride):
            layout_seed = split_base + i * stride + attempt
            inter_seed  = split_base + ROLE_OFFSET + i * stride + attempt
            try:
                sample = _generate_one(
                    layout_seed, inter_seed, itype, H, W, gamma, cfg,
                    composed_pair_idx=cidx,
                    sample_id=global_sample_id,
                    layout_id=layout_id,
                    nuisance_prob=nuisance_prob,
                    distractor_prob=distractor_prob,
                )
                samples.append(sample)
                success = True
                break
            except (ValueError, RuntimeError):
                pass

        if not success:
            failures += 1
            if verbose and failures <= 5:
                print(f"  [WARN] {split_name}[{i}] ({itype}): all retries exhausted, skipping")

        if verbose and (i + 1) % max(1, n // 10) == 0:
            elapsed = time.time() - t0
            print(f"  {split_name}: {i+1}/{n} | failures={failures} | {elapsed:.1f}s")

    if verbose:
        print(f"  {split_name}: done — {len(samples)}/{n} samples, {failures} failures")
    return samples


def _balanced_schedule(n, types, rng):
    per = n // len(types)
    rem = n % len(types)
    sched = []
    for t in types:
        sched.extend([t] * per)
    sched.extend(rng.choice(types, size=rem, replace=False).tolist())
    rng.shuffle(sched)
    return sched


def _balanced_pair_schedule(n, pair_ids, rng):
    """Build a shuffled schedule balanced across the requested composed pairs."""
    pair_ids = list(pair_ids)
    if not pair_ids:
        raise ValueError("composed split requires at least one pair id")
    per = n // len(pair_ids)
    rem = n % len(pair_ids)
    sched = []
    for pair_id in pair_ids:
        sched.extend([pair_id] * per)
    sched.extend(pair_ids[:rem])
    return [sched[i] for i in rng.permutation(len(sched))]


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_split(samples, path):
    if not samples:
        print(f"  [WARN] Nothing to save for {path}")
        return
    arrays = {k: np.array([s[k] for s in samples]) for k in samples[0]}
    np.savez_compressed(path, **arrays)


def load_split(path):
    return np.load(path, allow_pickle=True)


# ---------------------------------------------------------------------------
# Main generation entry point
# ---------------------------------------------------------------------------

def generate_dataset(
    out_dir, seed=0, H=10, W=10,
    n_train=12000, n_val=2000, n_test=2000, n_composed=2000,
    n_test_larger=0, n_test_noisy=0,
    n_train_composed_seen=0, n_val_composed_seen=0,
    n_test_composed_seen=0, n_test_composed_heldout=0,
    heldout_composed_pair_id=2,
    gamma=0.95, verbose=True,
):
    os.makedirs(out_dir, exist_ok=True)
    cfg = {**CONFIG, "gamma": gamma}
    master_rng = np.random.default_rng(seed)
    base_seed = seed * BASE_SEED_STRIDE
    t_start = time.time()

    splits = {}
    global_id = 0  # monotonic sample_id counter across splits

    all_pair_ids = list(range(len(COMPOSED_PAIRS)))
    if heldout_composed_pair_id not in all_pair_ids:
        raise ValueError(
            f"heldout_composed_pair_id={heldout_composed_pair_id} is out of range")
    seen_composed_pair_ids = [
        pair_id for pair_id in all_pair_ids
        if pair_id != heldout_composed_pair_id
    ]

    # (name, n, composed_pair_ids-or-None, H, W, nuisance_prob, distractor_prob)
    spec = [
        ("train_single",          n_train,                    None, H,  W,  0.15, 0.10),
        ("val_single",            n_val,                      None, H,  W,  0.15, 0.10),
        ("test_single",           n_test,                     None, H,  W,  0.15, 0.10),
        ("test_composed",         n_composed,                 all_pair_ids, H, W, 0.15, 0.10),
        ("test_larger",           n_test_larger,              None, 12, 12, 0.15, 0.10),
        ("test_noisy",            n_test_noisy,               None, H,  W,  0.35, 0.25),
        ("train_composed_seen",   n_train_composed_seen,      seen_composed_pair_ids, H, W, 0.15, 0.10),
        ("val_composed_seen",     n_val_composed_seen,        seen_composed_pair_ids, H, W, 0.15, 0.10),
        ("test_composed_seen",    n_test_composed_seen,       seen_composed_pair_ids, H, W, 0.15, 0.10),
        ("test_composed_heldout", n_test_composed_heldout,    [heldout_composed_pair_id], H, W, 0.15, 0.10),
    ]

    for split_name, n, composed_pair_ids, Hs, Ws, nz, dz in spec:
        if n <= 0:
            splits[split_name] = []
            continue
        split_id = SPLIT_IDS[split_name]
        print(f"Generating {split_name} ({n} samples, grid={Hs}x{Ws}, split_id={split_id})...")

        if composed_pair_ids is not None:
            pair_sched_raw = _balanced_pair_schedule(n, composed_pair_ids, master_rng)
            itype_sched = ["composed"] * len(pair_sched_raw)
        else:
            itype_sched = _balanced_schedule(n, SINGLE_INTERVENTIONS, master_rng)
            pair_sched_raw = None

        samples = generate_split(
            n, itype_sched, split_name, split_id, base_seed, Hs, Ws, gamma, cfg, verbose,
            composed_pair_schedule=pair_sched_raw,
            global_id_offset=global_id,
            nuisance_prob=nz, distractor_prob=dz,
        )
        splits[split_name] = samples
        global_id += n

    # Save
    print("Saving splits...")
    for split_name, samples in splits.items():
        path = os.path.join(out_dir, f"{split_name}.npz")
        save_split(samples, path)
        print(f"  Saved {len(samples)} samples → {path}")

    metadata = {
        "intervention_names": {str(v): k for k, v in INTERVENTION_IDS.items()},
        "composed_pairs":     [list(p) for p in COMPOSED_PAIRS],
        "heldout_composed_pair_id": int(heldout_composed_pair_id),
        "seen_composed_pair_ids": [int(x) for x in seen_composed_pair_ids],
        "composed_split_pair_ids": {
            "test_composed": [int(x) for x in all_pair_ids],
            "train_composed_seen": [int(x) for x in seen_composed_pair_ids],
            "val_composed_seen": [int(x) for x in seen_composed_pair_ids],
            "test_composed_seen": [int(x) for x in seen_composed_pair_ids],
            "test_composed_heldout": [int(heldout_composed_pair_id)],
        },
        "target_labels":      ["sensory_update", "value_remap", "map_remap", "action_remap"],
        "channels": {
            "0": "wall", "1": "start", "2": "goal",
            "3": "nuisance_visual", "4": "distractor",
            "5": "door_or_path_marker",
            "6": "one_way_up", "7": "one_way_down",
            "8": "one_way_left", "9": "one_way_right",
        },
        "scalar_errors": {
            "nuisance_error":    "mean L1 diff of nuisance+distractor channels",
            "full_visual_error": "mean L1 diff of all 10 grid channels",
            "sensory_error":     "alias of nuisance_error (backward compat)",
            "future_error":      "mean |delta_future|",
            "value_error":       "mean |delta_value|",
            "action_error":      "fraction of free (action,cell) pairs where transition changed",
            "action_changed_count": "count of changed free action transitions",
            "action_changed_cell_count": "count of free cells with at least one changed action transition",
            "action_error_local": "changed transitions divided by possible actions at changed cells",
        },
        "future_policy": "uniform attempted actions (invalid → stay); not uniform valid actions",
        "action_change_meta": {
            "action_cell_row": "row of one-way cell (-1 if not action_change)",
            "action_cell_col": "col of one-way cell (-1 if not action_change)",
            "action_cell_on_path": "1 if one-way cell lies on original shortest path",
            "action_cell_near_path": "1 if within radius-2 of original shortest path",
            "action_path_action_changed": "1 if one-way dir differs from original path action",
            "action_path_len_changed": "1 if directed path length changed by >=1",
        },
        "config": {k: (int(v) if isinstance(v, np.integer) else v) for k, v in cfg.items()},
        "seed": int(seed),
        "grid_H": H, "grid_W": W, "gamma": gamma,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "requested_split_sizes": {
            "train_single":  int(n_train),
            "val_single":    int(n_val),
            "test_single":   int(n_test),
            "test_composed": int(n_composed),
            "test_larger":   int(n_test_larger),
            "test_noisy":    int(n_test_noisy),
            "train_composed_seen":   int(n_train_composed_seen),
            "val_composed_seen":     int(n_val_composed_seen),
            "test_composed_seen":    int(n_test_composed_seen),
            "test_composed_heldout": int(n_test_composed_heldout),
        },
        "min_saved_fraction_recommended": 0.95,
        "split_ids": SPLIT_IDS,
        "layout_id_scheme": (
            f"layout_id = split_id * {LAYOUT_ID_STRIDE} + sample_index; "
            "provably disjoint across splits (independent of retry attempt)."
        ),
        "seed_scheme": (
            f"base_seed = seed * {BASE_SEED_STRIDE}; "
            f"layout_seed = base_seed + split_id*{SPLIT_SEED_STRIDE} + i*max_sample_tries + attempt; "
            f"intervention_seed = layout_seed + {ROLE_OFFSET}."
        ),
        "seed_offsets_DEPRECATED": SEED_OFFSETS,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Total time: {time.time() - t_start:.1f}s")
    return splits


def main():
    parser = argparse.ArgumentParser(description="Generate RemapBench dataset")
    parser.add_argument("--out",            default="data/remapbench_v0")
    parser.add_argument("--seed",           type=int, default=0)
    parser.add_argument("--grid_size",      type=int, default=10)
    parser.add_argument("--n_train",        type=int, default=12000)
    parser.add_argument("--n_val",          type=int, default=2000)
    parser.add_argument("--n_test",         type=int, default=2000)
    parser.add_argument("--n_composed",     type=int, default=2000)
    parser.add_argument("--n_test_larger",  type=int, default=0)
    parser.add_argument("--n_test_noisy",   type=int, default=0)
    parser.add_argument("--n_train_composed_seen",   type=int, default=0)
    parser.add_argument("--n_val_composed_seen",     type=int, default=0)
    parser.add_argument("--n_test_composed_seen",    type=int, default=0)
    parser.add_argument("--n_test_composed_heldout", type=int, default=0)
    parser.add_argument("--heldout_composed_pair_id", type=int, default=2)
    parser.add_argument("--gamma",          type=float, default=0.95)
    args = parser.parse_args()
    generate_dataset(
        out_dir=args.out, seed=args.seed, H=args.grid_size, W=args.grid_size,
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        n_composed=args.n_composed, n_test_larger=args.n_test_larger,
        n_test_noisy=args.n_test_noisy,
        n_train_composed_seen=args.n_train_composed_seen,
        n_val_composed_seen=args.n_val_composed_seen,
        n_test_composed_seen=args.n_test_composed_seen,
        n_test_composed_heldout=args.n_test_composed_heldout,
        heldout_composed_pair_id=args.heldout_composed_pair_id,
        gamma=args.gamma,
    )


if __name__ == "__main__":
    main()
