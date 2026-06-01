"""Generate the minimal nine-job decoupled causal-patching config set."""
import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.final_common import FOLDS


ROOT = Path("configs/decoupled_patching")


def _base(data_dir, run_name, seed, fold_id, fold_name, heldout_pair_id):
    return {
        "data_dir": data_dir,
        "model": "decoupled_gated_erpm",
        "report_model_name": "decoupled_gated_erpm",
        "seed": seed,
        "run_name": run_name,
        "fold_id": fold_id,
        "fold_name": fold_name,
        "heldout_composed_pair_id": heldout_pair_id,
        "train_splits": ["train_single", "train_composed_seen"],
        "val_splits": ["val_single", "val_composed_seen"],
        "batch_size": 128,
        "epochs": 30,
        "lr": 0.001,
        "device": "auto",
        "future_weight": 1.0,
        "value_weight": 1.0,
        "action_weight": 1.0,
        "type_weight": 1.0,
        "future_after_weight": 1.0,
        "value_after_weight": 1.0,
        "gate_sparsity_weight": 0.0,
        "stability_weight": 0.0,
        "checkpoint_metric": "best_val_composed_seen_exact",
    }


def configs():
    rows = []
    for fold in FOLDS:
        for seed in (0, 1):
            name = f"fold{fold['fold_id']}_decoupled_seed{seed}"
            cfg = _base(
                f"data/final_decomp_gap/{fold['fold_name']}",
                f"decoupled_patching_{name}", seed, fold["fold_id"],
                fold["fold_name"], fold["heldout_pair_id"])
            rows.append((ROOT / "direct" / f"{name}.yaml", cfg))
    cfg = _base(
        "data/final_decomp_gap/seen_upper_bound",
        "decoupled_patching_seen_upper_bound_seed0", 0, "seen_upper_bound",
        "seen_upper_bound", 2)
    cfg["include_all_composed_pairs_train"] = True
    rows.append((ROOT / "seen_upper_bound_seed0.yaml", cfg))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for path, cfg in configs():
        print(f"{path}: seed={cfg['seed']} data={cfg['data_dir']}")
        if not args.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"{'Would generate' if args.dry_run else 'Generated'} {len(configs())} configs")


if __name__ == "__main__":
    main()
