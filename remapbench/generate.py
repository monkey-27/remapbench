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
    "action_change":    {"action_error": 0.012},   # > validate THRESH_HIGH=0.01
}

SINGLE_INTERVENTIONS = ["sensory_nuisance", "goal_relocation", "topology_change", "action_change"]

# Seed offsets per split — must be spaced > max(n_train, n_val, n_test, n_composed) * max_retries
SEED_OFFSETS = {
    "train_single":   0,
    "val_single":     500_000,
    "test_single":    1_000_000,
    "test_composed":  1_500_000,
}


# ---------------------------------------------------------------------------
# Core sample generation
# ---------------------------------------------------------------------------

def _generate_one(
    layout_seed, intervention_seed, intervention_type, H, W, gamma, cfg,
    composed_pair_idx=None, sample_id=0,
):
    """Generate one sample with fully deterministic, per-sample seeds."""
    rng_layout = np.random.default_rng(layout_seed)
    rng_inter  = np.random.default_rng(intervention_seed)

    grid, start, goal, _ = make_grid(
        H, W, rng_layout,
        wall_prob=cfg["wall_prob"],
        min_path=cfg["min_path"],
    )

    if intervention_type == "composed":
        result = apply_intervention(grid, start, goal, rng_inter, "composed", composed_pair_idx)
        g2, g2_goal, iname, multihot, pair, weak = result
        pair_id = COMPOSED_PAIRS.index(pair) if pair in COMPOSED_PAIRS else -1
    else:
        result = apply_intervention(grid, start, goal, rng_inter, intervention_type)
        g2, g2_goal, iname, multihot, weak = result
        pair_id = -1

    iid = INTERVENTION_IDS.get(iname, INTERVENTION_IDS["composed"])
    sample = build_sample_arrays(
        grid, g2, start, goal, g2_goal,
        iid, multihot, gamma,
        layout_id=layout_seed,
        sample_id=sample_id,
        layout_seed=layout_seed,
        intervention_seed=intervention_seed,
        intervention_pair_id=pair_id,
        weak_action_change=weak,
    )

    # Post-hoc oracle threshold check
    thresholds = ORACLE_THRESHOLDS.get(intervention_type, {})
    for err_key, min_val in thresholds.items():
        if float(sample[err_key]) < min_val:
            raise ValueError(
                f"{intervention_type}: {err_key}={float(sample[err_key]):.4f} < {min_val}"
            )

    return sample


def generate_split(
    n, intervention_schedule, split_name, seed_offset, H, W, gamma, cfg, verbose=True,
    composed_pair_schedule=None, global_id_offset=0,
):
    """Generate n samples for a split. intervention_schedule is length-n list of type strings."""
    samples = []
    failures = 0
    t0 = time.time()

    for i, itype in enumerate(intervention_schedule):
        cidx = (composed_pair_schedule[i] if composed_pair_schedule else None)
        global_sample_id = global_id_offset + i
        success = False

        for attempt in range(cfg["max_sample_tries"]):
            layout_seed = seed_offset + i + attempt * 31337
            inter_seed  = seed_offset + i + attempt * 17 + 99999
            try:
                sample = _generate_one(
                    layout_seed, inter_seed, itype, H, W, gamma, cfg,
                    composed_pair_idx=cidx,
                    sample_id=global_sample_id,
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
    gamma=0.95, verbose=True,
):
    os.makedirs(out_dir, exist_ok=True)
    cfg = {**CONFIG, "gamma": gamma}
    master_rng = np.random.default_rng(seed)
    t_start = time.time()

    splits = {}
    global_id = 0  # monotonic sample_id counter across splits

    spec = [
        ("train_single",  n_train,   SEED_OFFSETS["train_single"]  + seed * 100, False),
        ("val_single",    n_val,     SEED_OFFSETS["val_single"]    + seed * 100, False),
        ("test_single",   n_test,    SEED_OFFSETS["test_single"]   + seed * 100, False),
        ("test_composed", n_composed,SEED_OFFSETS["test_composed"] + seed * 100, True),
    ]

    for split_name, n, seed_offset, is_composed in spec:
        if n <= 0:
            splits[split_name] = []
            continue
        print(f"Generating {split_name} ({n} samples)...")

        if is_composed:
            n_pairs = len(COMPOSED_PAIRS)
            per = n // n_pairs
            rem = n % n_pairs
            pair_sched_raw = []
            for pi in range(n_pairs):
                pair_sched_raw.extend([pi] * per)
            pair_sched_raw.extend(list(range(rem)))
            pair_sched_raw = [pair_sched_raw[i] for i in master_rng.permutation(len(pair_sched_raw))]
            itype_sched = ["composed"] * len(pair_sched_raw)
        else:
            itype_sched = _balanced_schedule(n, SINGLE_INTERVENTIONS, master_rng)
            pair_sched_raw = None

        samples = generate_split(
            n, itype_sched, split_name, seed_offset, H, W, gamma, cfg, verbose,
            composed_pair_schedule=pair_sched_raw,
            global_id_offset=global_id,
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
        },
        "future_policy": "uniform attempted actions (invalid → stay); not uniform valid actions",
        "config": {k: (int(v) if isinstance(v, np.integer) else v) for k, v in cfg.items()},
        "seed": int(seed),
        "grid_H": H, "grid_W": W, "gamma": gamma,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "seed_offsets": SEED_OFFSETS,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Total time: {time.time() - t_start:.1f}s")
    return splits


def main():
    parser = argparse.ArgumentParser(description="Generate RemapBench dataset")
    parser.add_argument("--out",       default="data/remapbench_v0")
    parser.add_argument("--seed",      type=int, default=0)
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--n_train",   type=int, default=12000)
    parser.add_argument("--n_val",     type=int, default=2000)
    parser.add_argument("--n_test",    type=int, default=2000)
    parser.add_argument("--n_composed",type=int, default=2000)
    parser.add_argument("--gamma",     type=float, default=0.95)
    args = parser.parse_args()
    generate_dataset(
        out_dir=args.out, seed=args.seed, H=args.grid_size, W=args.grid_size,
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        n_composed=args.n_composed, gamma=args.gamma,
    )


if __name__ == "__main__":
    main()
