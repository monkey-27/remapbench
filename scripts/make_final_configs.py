"""Generate deterministic configs for the 32-job decomposition-gap final run."""
import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.final_common import FOLDS


ROOT = Path("configs/final_decomp_gap")
DIRECT_MODELS = [
    ("standard", "standard_cnn"),
    ("gated", "gated_erpm"),
    ("independent_gate_heads", "factorized_gates"),
]
SEEN_MODELS = [("standard", "standard_cnn"), ("gated", "gated_erpm")]
SEEDS = [0, 1]


def _base(data_dir, run_name, model, report_model_name, seed, fold=None):
    config = {
        "data_dir": data_dir,
        "model": model,
        "report_model_name": report_model_name,
        "seed": seed,
        "run_name": run_name,
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
    if fold:
        config.update(
            fold_id=fold["fold_id"],
            fold_name=fold["fold_name"],
            heldout_composed_pair_id=fold["heldout_pair_id"],
        )
    return config


def configs():
    rows = []
    for fold in FOLDS:
        data_dir = f"data/final_decomp_gap/{fold['fold_name']}"
        for label, model in DIRECT_MODELS:
            for seed in SEEDS:
                name = f"fold{fold['fold_id']}_{label}_seed{seed}"
                rows.append((ROOT / "direct" / f"{name}.yaml", _base(
                    data_dir, f"final_direct_{name}", model, label, seed, fold)))
        name = f"fold{fold['fold_id']}_cpo_seed0"
        cfg = _base(data_dir, f"final_tuple_{name}", "cpo", "cpo_oracle_tuple", 0, fold)
        cfg.update(
            tuple_train_split="train_tuple_seen",
            tuple_val_split="val_tuple_seen",
            tuple_batch_size=32,
            gate_or_weight=1.0,
            operator_comp_weight=1.0,
            operator_reuse_weight=0.5,
            commutativity_weight=0.05,
            interaction_residual_weight=0.0,
            checkpoint_metric="best_val_tuple_loss",
        )
        rows.append((ROOT / "tuple" / f"{name}.yaml", cfg))

    for label, model in SEEN_MODELS:
        for seed in SEEDS:
            name = f"{label}_seed{seed}"
            cfg = _base("data/final_decomp_gap/seen_upper_bound",
                        f"final_seen_upper_bound_{name}", model, label, seed)
            cfg.update(
                fold_id="seen_upper_bound",
                fold_name="seen_upper_bound",
                heldout_composed_pair_id=2,
                include_all_composed_pairs_train=True,
            )
            rows.append((ROOT / "seen_upper_bound" / f"{name}.yaml", cfg))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = configs()
    for path, config in rows:
        print(f"{path}: model={config['model']} seed={config['seed']} data={config['data_dir']}")
        if not args.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Generated {len(rows)} configs" if not args.dry_run else f"Would generate {len(rows)} configs")


if __name__ == "__main__":
    main()
