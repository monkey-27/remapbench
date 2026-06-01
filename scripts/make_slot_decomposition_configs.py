"""Generate the exact eleven-job slot-decomposition pilot config set."""
import argparse
from pathlib import Path

import yaml


ROOT = Path("configs/slot_decomposition_pilot")
DATA_DIRS = {
    "fold0": "data/final_decomp_gap/fold0_goal_topology",
    "fold2": "data/final_decomp_gap/fold2_goal_action",
    "seen": "data/final_decomp_gap/seen_upper_bound",
}
FOLDS = {
    "fold0": ("fold0_goal_topology", 0),
    "fold2": ("fold2_goal_action", 2),
    "seen": ("seen_upper_bound", 2),
}
SEEDS = (0, 1)
ABLATIONS = ("no_component", "no_contrastive", "shared_pooled")


def _config(dataset, seed, variant="full"):
    fold_name, heldout_pair_id = FOLDS[dataset]
    name = f"{dataset}_{variant}_seed{seed}"
    config = {
        "data_dir": DATA_DIRS[dataset],
        "model": "slot_decomposition",
        "report_model_name": f"slot_decomposition_{variant}",
        "variant": variant,
        "seed": seed,
        "run_name": f"slot_decomposition_pilot_{name}",
        "fold_id": "seen_upper_bound" if dataset == "seen" else int(dataset[-1]),
        "fold_name": fold_name,
        "heldout_pair_id": heldout_pair_id,
        "heldout_composed_pair_id": heldout_pair_id,
        "train_splits": ["train_single", "train_composed_seen"],
        "val_splits": ["val_single", "val_composed_seen"],
        "batch_size": 128,
        "tuple_batch_size": 32,
        "lr": 0.001,
        "device": "auto",
        "epochs": 30,
        "stage1_epochs": 8,
        "stage2_epochs": 12,
        "stage3_epochs": 10,
        "checkpoint_metric": "best_val_composed_seen_exact",
        "future_weight": 1.0,
        "value_weight": 1.0,
        "action_weight": 1.0,
        "type_weight": 1.0,
        "label_weight": 1.0,
        "global_residual_weight": 1.0,
        "inactive_weight": 0.05,
        "diversity_weight": 0.01,
        "future_after_weight": 0.1,
        "value_after_weight": 0.1,
        "component_residual_weight": 0.5,
        "contrastive_weight": 0.2,
        "shared_pooled": variant == "shared_pooled",
    }
    if dataset != "seen":
        config.update(
            tuple_train_split="train_tuple_seen",
            tuple_val_split="val_tuple_seen",
            tuple_test_split="test_tuple_heldout",
        )
    else:
        config["include_all_composed_pairs_train"] = True
    if variant == "no_component":
        config["component_residual_weight"] = 0.0
    elif variant == "no_contrastive":
        config["contrastive_weight"] = 0.0
    return config


def configs():
    rows = []
    for dataset in ("fold0", "fold2"):
        for seed in SEEDS:
            cfg = _config(dataset, seed)
            rows.append((ROOT / f"{dataset}_full_seed{seed}.yaml", cfg))
    cfg = _config("seen", 0)
    rows.append((ROOT / "seen_full_seed0.yaml", cfg))
    for variant in ABLATIONS:
        for dataset in ("fold0", "fold2"):
            cfg = _config(dataset, 0, variant)
            rows.append((ROOT / f"{dataset}_{variant}_seed0.yaml", cfg))
    assert len(rows) == 11
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = configs()
    for path, config in rows:
        print(f"{path}: run={config['run_name']} data={config['data_dir']}")
        if not args.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(config, sort_keys=False))
    verb = "Would generate" if args.dry_run else "Generated"
    print(f"{verb} {len(rows)} configs")


if __name__ == "__main__":
    main()
