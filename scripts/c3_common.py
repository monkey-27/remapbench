"""Shared helpers for the C3 subset-search pilot."""
import os

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from models import build_model
from remapbench.data import RemapDataset


class OrdinaryRows(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = dict(self.dataset[idx])
        for key in ("tuple_id", "tuple_role_id", "tuple_pair_id"):
            row.pop(key, None)
        return row


def load_splits(data_dir, names, max_samples=None):
    parts = []
    for name in names:
        path = os.path.join(data_dir, f"{name}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        ds = OrdinaryRows(RemapDataset(path))
        if max_samples is not None:
            ds = Subset(ds, range(min(max_samples, len(ds))))
        parts.append(ds)
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def factor_targets(batch):
    return {key: batch[key] for key in ("delta_future", "delta_value", "action_delta")}


def build_c3_from_config(cfg):
    return build_model(
        cfg["model"],
        latent_channels=cfg.get("latent_channels", 32),
        sparsity_weight=cfg.get("sparsity_weight", 0.001),
        energy_scale=cfg.get("energy_scale", 100.0),
    )


def make_loader(dataset, cfg, shuffle=False):
    return DataLoader(dataset, batch_size=cfg.get("batch_size", 128), shuffle=shuffle,
                      num_workers=cfg.get("num_workers", 0))
