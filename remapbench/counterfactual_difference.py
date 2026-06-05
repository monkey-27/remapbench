"""Datasets and diagnostics for primitive evidence-mask training."""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from remapbench.data import RemapDataset, TupleRemapDataset


CAUSES = ("sensory", "value", "map", "action")
ROLE_TO_ID = {
    "base_to_A": 0,
    "base_to_B": 1,
    "A_to_AB": 2,
    "base_to_AB": 3,
}
PRIMITIVE_CHANNELS = {
    0: (3, 4),
    1: (2,),
    2: (0, 5),
    3: (6, 7, 8, 9),
}


def primitive_masks(before_grid, after_grid):
    diff = (after_grid - before_grid).abs()
    masks = []
    for channels in PRIMITIVE_CHANNELS.values():
        masks.append((diff[list(channels)].sum(dim=0) > 0).float())
    return torch.stack(masks, dim=0)


def transition(before_row, after_row, target, role, component_added=None):
    before = before_row["after_grid"]
    after = after_row["after_grid"]
    target = target.float()
    return {
        "before_grid": before,
        "after_grid": after,
        "x": torch.cat([before, after], dim=0),
        "x_diff": torch.cat([before, after, after - before, (after - before).abs()], dim=0),
        "target_multihot": target,
        "primitive_masks": primitive_masks(before, after),
        "transition_role_id": torch.tensor(ROLE_TO_ID[role], dtype=torch.int64),
        "transition_role_name": role,
        "component_added": target if component_added is None else component_added.float(),
        "tuple_id": before_row.get("tuple_id", torch.tensor(-1, dtype=torch.int64)),
        "pair_id": before_row.get("tuple_pair_id", torch.tensor(-1, dtype=torch.int64)),
        "layout_id": before_row.get("layout_id", torch.tensor(-1, dtype=torch.int64)),
        "sample_id": after_row.get("sample_id", torch.tensor(-1, dtype=torch.int64)),
    }


def tuple_transitions(item):
    a_target = item["A"]["target_multihot"]
    b_target = item["B"]["target_multihot"]
    ab_target = torch.clamp(a_target + b_target, max=1.0)
    return {
        "base_to_A": transition(item["base"], item["A"], a_target, "base_to_A"),
        "base_to_B": transition(item["base"], item["B"], b_target, "base_to_B"),
        "A_to_AB": transition(item["A"], item["AB"], b_target, "A_to_AB"),
        "base_to_AB": transition(item["base"], item["AB"], ab_target, "base_to_AB"),
    }


class CounterfactualTupleDataset(Dataset):
    """One item is a linked base/A/B/AB tuple expanded into named transitions."""

    def __init__(self, path, max_tuples=None):
        self.tuples = TupleRemapDataset(path)
        self.max_tuples = min(max_tuples, len(self.tuples)) if max_tuples else len(self.tuples)

    def __len__(self):
        return self.max_tuples

    def __getitem__(self, idx):
        item = self.tuples[idx]
        return {"tuple_id": item["tuple_id"], **tuple_transitions(item)}


class CounterfactualTransitionDataset(Dataset):
    """Flatten tuple transitions, or wrap an ordinary RemapBench split as transitions."""

    def __init__(self, path, roles=None, max_items=None):
        self.path = path
        self.roles = roles or tuple(ROLE_TO_ID)
        self.max_items = max_items
        with np.load(path, allow_pickle=True) as raw:
            self.is_tuple = "tuple_id" in raw.files and "tuple_role_id" in raw.files
        if self.is_tuple:
            self.source = CounterfactualTupleDataset(path)
            self.index = [(i, role) for i in range(len(self.source)) for role in self.roles]
        else:
            self.source = RemapDataset(path)
            self.index = [(i, None) for i in range(len(self.source))]
        if max_items is not None:
            self.index = self.index[:max_items]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        source_idx, role = self.index[idx]
        if self.is_tuple:
            return self.source[source_idx][role]
        row = self.source[source_idx]
        pair_id = row.get("intervention_pair_id", torch.tensor(-1, dtype=torch.int64))
        before = {
            "after_grid": row["before_grid"], "layout_id": row["layout_id"],
            "sample_id": row["sample_id"], "tuple_pair_id": pair_id,
        }
        after = {
            "after_grid": row["after_grid"], "layout_id": row["layout_id"],
            "sample_id": row["sample_id"], "tuple_pair_id": pair_id,
        }
        return transition(before, after, row["target_multihot"], "base_to_AB")


def split_path(data_dir, split):
    return os.path.join(data_dir, f"{split}.npz")
