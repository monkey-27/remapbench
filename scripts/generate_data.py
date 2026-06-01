"""
Generate RemapBench dataset from a config YAML.
Usage:
    python scripts/generate_data.py --config configs/smoke.yaml
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from remapbench.generate import generate_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    H = W = cfg.get("grid_size", 10)
    generate_dataset(
        out_dir=cfg["data_dir"],
        seed=cfg.get("seed", 0),
        H=H, W=W,
        n_train=cfg.get("n_train", 12000),
        n_val=cfg.get("n_val", 2000),
        n_test=cfg.get("n_test", 2000),
        n_composed=cfg.get("n_composed", 2000),
        n_test_larger=cfg.get("n_test_larger", 0),
        n_test_noisy=cfg.get("n_test_noisy", 0),
        n_train_composed_seen=cfg.get("n_train_composed_seen", 0),
        n_val_composed_seen=cfg.get("n_val_composed_seen", 0),
        n_test_composed_seen=cfg.get("n_test_composed_seen", 0),
        n_test_composed_heldout=cfg.get("n_test_composed_heldout", 0),
        n_train_tuple_seen=cfg.get("n_train_tuple_seen", 0),
        n_val_tuple_seen=cfg.get("n_val_tuple_seen", 0),
        n_test_tuple_seen=cfg.get("n_test_tuple_seen", 0),
        n_test_tuple_heldout=cfg.get("n_test_tuple_heldout", 0),
        heldout_composed_pair_id=cfg.get("heldout_composed_pair_id", 2),
        include_all_composed_pairs_train=cfg.get("include_all_composed_pairs_train", False),
        gamma=cfg.get("gamma", 0.95),
    )


if __name__ == "__main__":
    main()
