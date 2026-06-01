"""Smoke-test decoupled routing gates, logits, and gradient connectivity."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from models.gated_erpm import _safe_logit


def _has_grad(module):
    return any(param.grad is not None and torch.isfinite(param.grad).all()
               and float(param.grad.abs().sum()) > 0 for param in module.parameters())


def main():
    torch.manual_seed(0)
    model = build_model("decoupled_gated_erpm")
    out = model(torch.randn(3, 20, 10, 10))
    expected = {
        "remap_logits", "routing_gate_logits", "routing_gates", "gates", "gate_logits",
        "z_before", "z_update", "z_updated", "pathway_updates", "future_after",
        "value_after", "delta_future", "delta_value", "action_delta",
    }
    missing = sorted(expected - set(out))
    if missing:
        raise AssertionError(f"Missing output keys: {missing}")
    patched_gates = out["gates"].detach().clone()
    patched_gates[:, 0] = 0.99
    patched = model.decode_from_components(out["z_before"], patched_gates, out["pathway_updates"])
    if torch.allclose(patched["remap_logits"], _safe_logit(patched_gates)):
        raise AssertionError("Patched remap logits still equal patched gate logits")
    loss = sum(value.square().mean() for value in (
        out["remap_logits"], out["future_after"], out["value_after"],
        out["delta_future"], out["delta_value"], out["action_delta"]))
    loss.backward()
    checks = {
        "gate_net": _has_grad(model.gate_net),
        "pathway_heads": all(_has_grad(head) for head in (
            model.sensory_head, model.value_head, model.map_head, model.action_head)),
        "remap_classifier": _has_grad(model.remap_classifier),
        "decoders": all(_has_grad(head) for head in (
            model.future_after_head, model.value_after_head, model.delta_future_head,
            model.delta_value_head, model.action_delta_head)),
    }
    if not all(checks.values()):
        raise AssertionError(f"Missing gradients: {checks}")
    print("Decoupled GatedERPM smoke: PASS")


if __name__ == "__main__":
    main()
