"""
Dataset generation CLI.
Usage:
    python -m remapbench.generate --out data/remapbench_v0 --seed 0 \
        --grid_size 10 --n_train 12000 --n_val 2000 --n_test 2000 --n_composed 2000
"""
import argparse
import json
import os
import sys
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
    "max_sample_tries": 50,   # rejection sampling retries per sample
    "max_layout_tries": 200,  # retries for grid generation
}

# Minimum oracle error required per intervention type (post-hoc validation)
ORACLE_THRESHOLDS = {
    "sensory_nuisance": {"sensory_error": 0.02},
    "goal_relocation":  {"value_error": 0.02},
    "topology_change":  {"future_error": 0.008},
    "action_change":    {"action_error": 0.012},
}

SINGLE_INTERVENTIONS = ["sensory_nuisance", "goal_relocation", "topology_change", "action_change"]

# Seed offsets per split to guarantee disjoint base layouts
SEED_OFFSETS = {
    "train_single":   0,
    "val_single":     500_000,
    "test_single":    1_000_000,
    "test_composed":  1_500_000,
}


# ---------------------------------------------------------------------------
# Core sample generation
# ---------------------------------------------------------------------------

def _generate_one(layout_seed, intervention_type, rng_inter, H, W, gamma, cfg, composed_pair_idx=None):
    """Generate one (layout, intervention) sample. Returns dict or raises ValueError."""
    rng_layout = np.random.default_rng(layout_seed)
    grid, start, goal, _ = make_grid(
        H, W, rng_layout,
        wall_prob=cfg["wall_prob"],
        min_path=cfg["min_path"],
    )

    if intervention_type == "composed":
        result = apply_intervention(grid, start, goal, rng_inter, "composed", composed_pair_idx)
        g2, g2_goal, iname, multihot = result[0], result[1], result[2], result[3]
    else:
        g2, g2_goal, iname, multihot = apply_intervention(
            grid, start, goal, rng_inter, intervention_type
        )

    iid = INTERVENTION_IDS.get(iname, INTERVENTION_IDS["composed"])
    sample = build_sample_arrays(grid, g2, start, goal, g2_goal, iid, multihot, gamma)

    # Post-hoc oracle threshold check (reject if signal too weak)
    thresholds = ORACLE_THRESHOLDS.get(intervention_type, {})
    for err_key, min_val in thresholds.items():
        if float(sample[err_key]) < min_val:
            raise ValueError(
                f"{intervention_type}: {err_key}={float(sample[err_key]):.4f} < {min_val}"
            )

    return sample


def generate_split(n, intervention_schedule, split_name, base_seed, H, W, gamma, cfg, verbose=True):
    """
    Generate n samples for a split.
    intervention_schedule: list of intervention_type strings (length n, balanced).
    Returns list of sample dicts.
    """
    samples = []
    failures = 0
    t0 = time.time()
    inter_rng = np.random.default_rng(base_seed + 999_999)

    for i, itype in enumerate(intervention_schedule):
        layout_seed = base_seed + i
        composed_idx = None
        if itype == "composed":
            composed_idx = i % len(COMPOSED_PAIRS)

        success = False
        for attempt in range(cfg["max_sample_tries"]):
            try:
                sample = _generate_one(
                    layout_seed + attempt * 7,  # vary layout seed on retry
                    itype, inter_rng, H, W, gamma, cfg, composed_idx
                )
                samples.append(sample)
                success = True
                break
            except (ValueError, RuntimeError):
                pass

        if not success:
            failures += 1
            if verbose and failures <= 5:
                print(f"  [WARN] {split_name} sample {i} ({itype}): all retries failed, skipping")

        if verbose and (i + 1) % max(1, n // 10) == 0:
            elapsed = time.time() - t0
            print(f"  {split_name}: {i+1}/{n} samples, {failures} failures, {elapsed:.1f}s")

    if verbose:
        print(f"  {split_name}: done — {len(samples)} samples, {failures} failures")
    return samples


def _balanced_schedule(n, types, rng):
    """Return list of n intervention types, balanced across types."""
    per_type = n // len(types)
    remainder = n % len(types)
    schedule = []
    for t in types:
        schedule.extend([t] * per_type)
    extra_types = rng.choice(types, size=remainder, replace=False).tolist()
    schedule.extend(extra_types)
    rng.shuffle(schedule)
    return schedule


def _balanced_composed_schedule(n, rng):
    """Return balanced schedule over composed pair types."""
    pair_names = [f"composed_{i}" for i in range(len(COMPOSED_PAIRS))]
    per = n // len(COMPOSED_PAIRS)
    remainder = n % len(COMPOSED_PAIRS)
    schedule = []
    for i in range(len(COMPOSED_PAIRS)):
        schedule.extend([(f"composed_{i}", i)] * per)
    for i in range(remainder):
        schedule.append((f"composed_{i}", i))
    idxs = rng.permutation(len(schedule))
    schedule = [schedule[i] for i in idxs]
    return schedule


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_split(samples, path):
    """Save list of sample dicts as a single compressed npz."""
    if not samples:
        print(f"  [WARN] No samples to save for {path}")
        return
    arrays = {}
    for key in samples[0]:
        arr = np.array([s[key] for s in samples])
        arrays[key] = arr
    np.savez_compressed(path, **arrays)


def load_split(path):
    """Load npz back to list of dicts (lazy — returns the npz object)."""
    return np.load(path, allow_pickle=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def generate_dataset(
    out_dir, seed=0, H=10, W=10,
    n_train=12000, n_val=2000, n_test=2000, n_composed=2000,
    gamma=0.95, verbose=True,
):
    os.makedirs(out_dir, exist_ok=True)
    cfg = CONFIG.copy()
    cfg["gamma"] = gamma

    master_rng = np.random.default_rng(seed)
    t_start = time.time()

    splits = {}

    # --- train_single ---
    if n_train > 0:
        print(f"Generating train_single ({n_train} samples)...")
        sched = _balanced_schedule(n_train, SINGLE_INTERVENTIONS, master_rng)
        splits["train_single"] = generate_split(
            n_train, sched, "train_single",
            SEED_OFFSETS["train_single"] + seed * 100,
            H, W, gamma, cfg, verbose
        )

    # --- val_single ---
    if n_val > 0:
        print(f"Generating val_single ({n_val} samples)...")
        sched = _balanced_schedule(n_val, SINGLE_INTERVENTIONS, master_rng)
        splits["val_single"] = generate_split(
            n_val, sched, "val_single",
            SEED_OFFSETS["val_single"] + seed * 100,
            H, W, gamma, cfg, verbose
        )

    # --- test_single ---
    if n_test > 0:
        print(f"Generating test_single ({n_test} samples)...")
        sched = _balanced_schedule(n_test, SINGLE_INTERVENTIONS, master_rng)
        splits["test_single"] = generate_split(
            n_test, sched, "test_single",
            SEED_OFFSETS["test_single"] + seed * 100,
            H, W, gamma, cfg, verbose
        )

    # --- test_composed ---
    if n_composed > 0:
        print(f"Generating test_composed ({n_composed} samples)...")
        comp_sched_raw = _balanced_composed_schedule(n_composed, master_rng)
        # Unpack into flat type list + store pair indices for _generate_one
        comp_sched = ["composed"] * len(comp_sched_raw)
        comp_pair_idxs = [ci for (_, ci) in comp_sched_raw]

        samples_c = []
        inter_rng = np.random.default_rng(SEED_OFFSETS["test_composed"] + seed * 100 + 999_999)
        failures = 0
        t0 = time.time()
        for i, (itype, cidx) in enumerate(zip(comp_sched, comp_pair_idxs)):
            layout_seed = SEED_OFFSETS["test_composed"] + seed * 100 + i
            success = False
            for attempt in range(cfg["max_sample_tries"]):
                try:
                    sample = _generate_one(
                        layout_seed + attempt * 7, "composed", inter_rng,
                        H, W, gamma, cfg, cidx
                    )
                    samples_c.append(sample)
                    success = True
                    break
                except (ValueError, RuntimeError):
                    pass
            if not success:
                failures += 1
            if verbose and (i + 1) % max(1, n_composed // 10) == 0:
                print(f"  test_composed: {i+1}/{n_composed} samples, {failures} failures, {time.time()-t0:.1f}s")
        if verbose:
            print(f"  test_composed: done — {len(samples_c)} samples, {failures} failures")
        splits["test_composed"] = samples_c

    # Save all splits
    print("Saving splits...")
    for split_name, samples in splits.items():
        path = os.path.join(out_dir, f"{split_name}.npz")
        save_split(samples, path)
        print(f"  Saved {len(samples)} samples → {path}")

    # Intervention name → id mapping
    metadata = {
        "intervention_names": {str(v): k for k, v in INTERVENTION_IDS.items()},
        "composed_pairs": [list(p) for p in COMPOSED_PAIRS],
        "target_labels": ["no_remap", "value_remap", "map_remap", "action_remap"],
        "channels": {
            "0": "wall", "1": "start", "2": "goal",
            "3": "nuisance_visual", "4": "distractor", "5": "door_or_path_marker",
            "6": "one_way_up", "7": "one_way_down",
            "8": "one_way_left", "9": "one_way_right",
        },
        "config": {k: (v if not isinstance(v, np.integer) else int(v)) for k, v in cfg.items()},
        "seed": int(seed),
        "grid_H": H,
        "grid_W": W,
        "gamma": gamma,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "seed_offsets": SEED_OFFSETS,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    total = time.time() - t_start
    print(f"\nDone. Total time: {total:.1f}s")
    return splits


def main():
    parser = argparse.ArgumentParser(description="Generate RemapBench dataset")
    parser.add_argument("--out", default="data/remapbench_v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--n_train", type=int, default=12000)
    parser.add_argument("--n_val", type=int, default=2000)
    parser.add_argument("--n_test", type=int, default=2000)
    parser.add_argument("--n_composed", type=int, default=2000)
    parser.add_argument("--gamma", type=float, default=0.95)
    args = parser.parse_args()

    H = W = args.grid_size
    generate_dataset(
        out_dir=args.out,
        seed=args.seed,
        H=H, W=W,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        n_composed=args.n_composed,
        gamma=args.gamma,
    )


if __name__ == "__main__":
    main()
