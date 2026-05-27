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
        gamma=cfg.get("gamma", 0.95),
    )


if __name__ == "__main__":
    main()
