"""Fast local checks for the C3 subset-search pilot."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from models import build_model
from models.c3 import enumerate_cause_subsets


VARIANTS = ("c3_full_reconstruction", "c3_delta_space", "c3_factor_scored")


def _fake_batch():
    batch = 3
    target = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 1], [1, 1, 1, 1]], dtype=torch.float32)
    return {
        "before_grid": torch.rand(batch, 10, 8, 8),
        "after_grid": torch.rand(batch, 10, 8, 8),
        "delta_future": torch.rand(batch, 1, 8, 8),
        "delta_value": torch.rand(batch, 1, 8, 8),
        "action_delta": torch.rand(batch, 4, 8, 8),
        "target_multihot": target,
    }


def _toy_recovery():
    subsets = enumerate_cause_subsets()
    true_id = 5
    observed = subsets[true_id]
    scores = (subsets - observed).square().sum(dim=1)
    assert int(scores.argmin()) == true_id


def _heldout_absent(data_dir):
    raw = np.load(os.path.join(data_dir, "train_composed_seen.npz"))
    pair_ids = set(int(value) for value in raw["intervention_pair_id"])
    assert 2 not in pair_ids, f"held-out pair 2 leaked into {data_dir}: {pair_ids}"


def main():
    subsets = enumerate_cause_subsets()
    assert subsets.shape == (16, 4)
    assert len({tuple(row.tolist()) for row in subsets}) == 16
    _toy_recovery()
    for data_dir in ("data/remapbench_v1_composed", "data/remapbench_cpo_pilot"):
        _heldout_absent(data_dir)
    for variant in VARIANTS:
        model = build_model(variant, latent_channels=8)
        losses, out = model.loss(_fake_batch())
        losses["total"].backward()
        assert out["scores"].shape == (3, 16)
        assert out["predicted_multihot"].shape == (3, 4)
        assert any(parameter.grad is not None for parameter in model.parameters())
        print(f"PASS {variant}: loss={float(losses['total']):.4f}")
    print("PASS subset enumeration: exactly 16 unique candidates")
    print("PASS toy recovery: known subset is argmin")
    print("PASS fairness: held-out pair 2 absent from D1/D2 train_composed_seen")


if __name__ == "__main__":
    main()
