"""Generate the compact D1/D2 C3 pilot config grid."""
import os

import yaml


DATASETS = {"d1": "data/remapbench_v1_composed", "d2": "data/remapbench_cpo_pilot"}
VARIANTS = ("full_reconstruction", "delta_space", "factor_scored")
SPARSITY_SETTINGS = ((0.0, "_sparse0"), (0.001, ""), (0.01, "_sparse01"))


def main():
    out_dir = os.path.join("configs", "c3_subset_search")
    os.makedirs(out_dir, exist_ok=True)
    for dataset, data_dir in DATASETS.items():
        for variant in VARIANTS:
            for sparsity_weight, suffix in SPARSITY_SETTINGS:
                cfg = {
                "data_dir": data_dir,
                "run_name": f"c3_{dataset}_{variant}{suffix}",
                "model": f"c3_{variant}",
                "train_splits": ["train_single", "train_composed_seen"],
                "val_splits": ["val_single", "val_composed_seen"],
                "epochs": 30,
                "batch_size": 128,
                "lr": 0.001,
                "device": "auto",
                "latent_channels": 32,
                "sparsity_weight": sparsity_weight,
                "energy_scale": 100.0,
            }
                path = os.path.join(out_dir, f"{dataset}_{variant}{suffix}.yaml")
                with open(path, "w") as handle:
                    yaml.safe_dump(cfg, handle, sort_keys=False)
                print(path)


if __name__ == "__main__":
    main()
