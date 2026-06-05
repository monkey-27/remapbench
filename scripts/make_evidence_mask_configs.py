"""Generate evidence-mask pilot configs."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from scripts.final_common import FOLDS


VARIANTS = {
    "bce_only": dict(
        primitive_mask_weight=0.0, inactive_weight=0.0,
        context_difference_weight=0.0, mixed_union_weight=0.0,
        use_diff_channels=True, label_from_evidence_pool=False,
    ),
    "bce_mask": dict(
        primitive_mask_weight=1.0, inactive_weight=0.0,
        context_difference_weight=0.0, mixed_union_weight=0.0,
        use_diff_channels=True, label_from_evidence_pool=False,
    ),
    "bce_mask_inactive": dict(
        primitive_mask_weight=1.0, inactive_weight=0.05,
        context_difference_weight=0.0, mixed_union_weight=0.0,
        use_diff_channels=True, label_from_evidence_pool=False,
    ),
    "bce_mask_inactive_context": dict(
        primitive_mask_weight=1.0, inactive_weight=0.05,
        context_difference_weight=0.2, mixed_union_weight=0.2,
        use_diff_channels=True, label_from_evidence_pool=False,
    ),
    "no_diff_channels": dict(
        primitive_mask_weight=1.0, inactive_weight=0.05,
        context_difference_weight=0.0, mixed_union_weight=0.0,
        use_diff_channels=False, label_from_evidence_pool=False,
    ),
    "label_from_evidence_maxpool": dict(
        primitive_mask_weight=1.0, inactive_weight=0.05,
        context_difference_weight=0.0, mixed_union_weight=0.0,
        use_diff_channels=True, label_from_evidence_pool=True,
    ),
}


def config_for(fold, variant, seed, smoke=False):
    name = "smoke" if smoke else f"{fold['fold_name']}_{variant}_seed{seed}"
    return {
        "model": "evidence_mask",
        "method": "evidence_mask",
        "loss_name": "primitive_mask_loss",
        "report_model_name": f"evidence_mask_{variant}",
        "fold_id": fold["fold_id"],
        "fold_name": fold["fold_name"],
        "heldout_pair_id": fold["heldout_pair_id"],
        "heldout_composed_pair_id": fold["heldout_pair_id"],
        "seed": seed,
        "data_dir": f"data/final_decomp_gap/{fold['fold_name']}",
        "tuple_train_split": "train_tuple_seen",
        "tuple_val_split": "val_tuple_seen",
        "tuple_test_split": "test_tuple_heldout",
        "val_split": "val_composed_seen",
        "epochs": 2 if smoke else 12,
        "batch_size": 256,
        "tuple_batch_size": 128,
        "hidden_channels": 48,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "label_weight": 1.0,
        "checkpoint_metric": "val_exact",
        "run_name": f"evidence_mask_{name}",
        **VARIANTS[variant],
    }


def main():
    out_dir = "configs/evidence_mask"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "smoke.yaml"), "w") as handle:
        yaml.safe_dump(config_for(FOLDS[0], "bce_mask_inactive", 0, smoke=True), handle, sort_keys=False)
    for fold in FOLDS:
        for seed in (0, 1):
            for variant in VARIANTS:
                cfg = config_for(fold, variant, seed)
                path = os.path.join(out_dir, f"{fold['fold_name']}_{variant}_seed{seed}.yaml")
                with open(path, "w") as handle:
                    yaml.safe_dump(cfg, handle, sort_keys=False)
    print(f"Wrote evidence-mask configs to {out_dir}")


if __name__ == "__main__":
    main()
