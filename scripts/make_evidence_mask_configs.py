"""Generate evidence-mask pilot configs."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from scripts.final_common import FOLDS


VARIANTS = {
    "learned_bce_only": dict(
        primitive_mask_weight=0.0, inactive_weight=0.0,
        context_difference_weight=0.0, mixed_union_weight=0.0,
        map_context_weight=0.0, map_union_weight=0.0,
        use_diff_channels=True, label_readout="learned_classifier",
    ),
    "learned_bce_mask": dict(
        primitive_mask_weight=1.0, inactive_weight=0.0,
        context_difference_weight=0.0, mixed_union_weight=0.0,
        map_context_weight=0.0, map_union_weight=0.0,
        use_diff_channels=True, label_readout="learned_classifier",
    ),
    "evidence_maxpool_bce_mask": dict(
        primitive_mask_weight=1.0, inactive_weight=0.0,
        context_difference_weight=0.0, mixed_union_weight=0.0,
        map_context_weight=0.0, map_union_weight=0.0,
        use_diff_channels=True, label_readout="evidence_maxpool",
    ),
    "evidence_logsumexp_bce_mask": dict(
        primitive_mask_weight=1.0, inactive_weight=0.0,
        context_difference_weight=0.0, mixed_union_weight=0.0,
        map_context_weight=0.0, map_union_weight=0.0,
        use_diff_channels=True, label_readout="evidence_logsumexp", readout_tau=0.5,
    ),
    "evidence_noisy_or_bce_mask": dict(
        primitive_mask_weight=1.0, inactive_weight=0.0,
        context_difference_weight=0.0, mixed_union_weight=0.0,
        map_context_weight=0.0, map_union_weight=0.0,
        use_diff_channels=True, label_readout="evidence_noisy_or",
    ),
    "logsumexp_mask_map_union": dict(
        primitive_mask_weight=1.0, inactive_weight=0.0,
        context_difference_weight=0.0, mixed_union_weight=0.0,
        map_context_weight=0.0, map_union_weight=0.5,
        use_diff_channels=True, label_readout="evidence_logsumexp", readout_tau=0.5,
    ),
    "logsumexp_mask_map_context": dict(
        primitive_mask_weight=1.0, inactive_weight=0.0,
        context_difference_weight=0.0, mixed_union_weight=0.0,
        map_context_weight=0.5, map_union_weight=0.0,
        use_diff_channels=True, label_readout="evidence_logsumexp", readout_tau=0.5,
    ),
    "logsumexp_mask_vector_context": dict(
        primitive_mask_weight=1.0, inactive_weight=0.0,
        context_difference_weight=0.2, mixed_union_weight=0.0,
        map_context_weight=0.0, map_union_weight=0.0,
        use_diff_channels=True, label_readout="evidence_logsumexp", readout_tau=0.5,
    ),
    "logsumexp_mask_vector_context_conflict_aware": dict(
        primitive_mask_weight=1.0, inactive_weight=0.0,
        context_difference_weight=0.2, mixed_union_weight=0.0,
        map_context_weight=0.0, map_union_weight=0.0,
        conflict_aware_mask_loss=True,
        conflict_mask_downweight=0.1,
        use_diff_channels=True, label_readout="evidence_logsumexp", readout_tau=0.5,
    ),
    "logsumexp_mask_map_context_union": dict(
        primitive_mask_weight=1.0, inactive_weight=0.0,
        context_difference_weight=0.0, mixed_union_weight=0.0,
        map_context_weight=0.5, map_union_weight=0.5,
        use_diff_channels=True, label_readout="evidence_logsumexp", readout_tau=0.5,
    ),
    "logsumexp_mask_context_union_inactive": dict(
        primitive_mask_weight=1.0, inactive_weight=0.05,
        context_difference_weight=0.0, mixed_union_weight=0.0,
        map_context_weight=0.5, map_union_weight=0.5,
        use_diff_channels=True, label_readout="evidence_logsumexp", readout_tau=0.5,
    ),
    "logsumexp_mask_union_conflict_aware": dict(
        primitive_mask_weight=1.0, inactive_weight=0.0,
        context_difference_weight=0.0, mixed_union_weight=0.0,
        map_context_weight=0.0, map_union_weight=0.5,
        conflict_aware_mask_loss=True,
        conflict_mask_downweight=0.1,
        use_diff_channels=True, label_readout="evidence_logsumexp", readout_tau=0.5,
    ),
    "no_diff_channels_logsumexp_mask_union": dict(
        primitive_mask_weight=1.0, inactive_weight=0.0,
        context_difference_weight=0.0, mixed_union_weight=0.0,
        map_context_weight=0.0, map_union_weight=0.5,
        use_diff_channels=False, label_readout="evidence_logsumexp", readout_tau=0.5,
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
        "run_name": f"evidence_mask_sieve_{name}",
        **VARIANTS[variant],
    }


def main():
    out_dir = "configs/evidence_mask"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "smoke.yaml"), "w") as handle:
        yaml.safe_dump(config_for(FOLDS[0], "evidence_logsumexp_bce_mask", 0, smoke=True), handle, sort_keys=False)
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
