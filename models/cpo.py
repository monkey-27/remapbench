"""Compositional pathway operators and their factorized-gate baseline."""
import torch
import torch.nn as nn

from .gated_erpm import GatedERPM, LATENT, N_GRID, _safe_logit


PATHWAYS = ("sensory", "value", "map", "action")


class _FactorizedGateHeads(nn.Module):
    """Four independent classifiers: no shared MLP parameters across gates."""

    def __init__(self, latent):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
            )
            for _ in PATHWAYS
        ])

    def forward(self, pooled):
        return torch.cat([head(pooled) for head in self.heads], dim=1)


class CPO(GatedERPM):
    """Gated ERPM with additive pathway-local operators for intervention tuples."""

    composition_losses_enabled = True

    def __init__(self, n_grid=N_GRID, latent=LATENT):
        super().__init__(n_grid=n_grid, latent=latent)

    def forward(self, x, gate_override=None, target_multihot=None):
        out = super().forward(
            x,
            gate_override=gate_override,
            target_multihot=target_multihot,
        )
        out["composition_diagnostics"] = {
            "pathway_update_rms": {
                name: update.square().mean().sqrt()
                for name, update in out["pathway_updates"].items()
            },
        }
        return out

    def forward_tuple(
        self,
        x_left,
        x_right,
        gate_override=None,
        target_multihot=None,
    ):
        """Compose two component examples through independent pathway operators."""
        left = self.forward(x_left, gate_override=gate_override)
        right = self.forward(x_right, gate_override=gate_override)

        composed_gates = 1.0 - (1.0 - left["gates"]) * (1.0 - right["gates"])
        if gate_override is not None:
            composed_gates = self._apply_override(
                composed_gates, gate_override, target_multihot)

        pathway_updates = {}
        additive_updates = {}
        for index, name in enumerate(PATHWAYS):
            left_update = left["gates"][:, index:index + 1, None, None] * left["pathway_updates"][name]
            right_update = right["gates"][:, index:index + 1, None, None] * right["pathway_updates"][name]
            additive_updates[name] = left_update + right_update
            pathway_updates[name] = additive_updates[name]

        z_update = sum(pathway_updates.values())
        z_updated = left["z_before"] + z_update
        remap_logits = _safe_logit(composed_gates)
        residual_loss = z_update.sum() * 0.0

        return {
            "future_after": self.future_after_head(z_updated),
            "value_after": self.value_after_head(z_updated),
            "delta_future": self.delta_future_head(z_updated),
            "delta_value": self.delta_value_head(z_updated),
            "action_delta": self.action_delta_head(z_updated),
            "remap_logits": remap_logits,
            "raw_gate_logits": remap_logits,
            "raw_gates": composed_gates,
            "gates": composed_gates,
            "gate_logits": remap_logits,
            "z_before": left["z_before"],
            "z_after": right["z_after"],
            "z_updated": z_updated,
            "pathway_updates": pathway_updates,
            "tuple_composition_loss": residual_loss,
            "tuple_composition_diagnostics": {
                "left_gates": left["gates"],
                "right_gates": right["gates"],
                "additive_pathway_updates": additive_updates,
            },
        }


class FactorizedGates(GatedERPM):
    """Gated ERPM baseline with independent gate heads and no tuple operators."""

    composition_losses_enabled = False

    def __init__(self, n_grid=N_GRID, latent=LATENT):
        super().__init__(n_grid=n_grid, latent=latent)
        self.gate_net = _FactorizedGateHeads(latent)
