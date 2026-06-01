"""Fast forward/backward and linked-tuple checks for slot decomposition."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models import build_model
from scripts.train_slot_decomposition import _direct_loss, _tuple_loss


def _row(batch=3):
    return {
        "x": torch.rand(batch, 20, 8, 8),
        "delta_future": torch.rand(batch, 1, 8, 8),
        "delta_value": torch.rand(batch, 1, 8, 8),
        "action_delta": torch.rand(batch, 4, 8, 8),
        "future_after": torch.rand(batch, 1, 8, 8),
        "value_after": torch.rand(batch, 1, 8, 8),
        "target_multihot": torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]]),
    }


def main():
    cfg = {
        "label_weight": 1.0, "global_residual_weight": 1.0,
        "component_residual_weight": 0.5, "inactive_weight": 0.05,
        "contrastive_weight": 0.2, "diversity_weight": 0.01,
    }
    for shared in (False, True):
        model = build_model("slot_decomposition", latent=8, shared_pooled=shared)
        direct, _, out = _direct_loss(model, _row(), cfg)
        direct.backward()
        assert out["remap_logits"].shape == (3, 4)
        assert out["routing_logits"].shape == (3, 4)
        assert out["routing_weights"].shape == (3, 4)
        assert out["slot_vectors"].shape == (3, 4, 8)
        assert out["slot_attention"].shape == (3, 4, 8, 8)
        assert out["slot_residuals"].shape == (3, 4, 6, 8, 8)
        assert out["residual_total"].shape == (3, 6, 8, 8)
        assert out["remap_logits"].data_ptr() != out["routing_logits"].data_ptr()
        print(f"PASS direct shared_pooled={shared}: loss={float(direct):.4f}")
    model = build_model("slot_decomposition", latent=8)
    tuple_batch = {"tuple_id": torch.arange(3), "A": _row(), "B": _row(), "AB": _row()}
    tuple_batch["A"]["target_multihot"] = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.0]])
    tuple_batch["B"]["target_multihot"] = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0.0]])
    tuple_batch["AB"]["target_multihot"] = torch.maximum(
        tuple_batch["A"]["target_multihot"], tuple_batch["B"]["target_multihot"])
    loss, metrics = _tuple_loss(model, tuple_batch, cfg)
    loss.backward()
    assert metrics["component_residual_mse"] >= 0
    assert metrics["contrastive_loss"] >= 0
    print(f"PASS tuple loss: {metrics}")


if __name__ == "__main__":
    main()
