"""Gated ERPM with routing gates decoupled from remap classification logits."""
import torch
import torch.nn as nn

from .gated_erpm import GatedERPM, LATENT, N_GRID, _safe_logit


class DecoupledGatedERPM(GatedERPM):
    """Route latent updates with gates while classifying remaps from updated state."""

    def __init__(self, n_grid=N_GRID, latent=LATENT):
        super().__init__(n_grid=n_grid, latent=latent)
        self.remap_classifier = nn.Sequential(
            nn.Linear(latent, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 4),
        )

    def decode_from_components(self, z_before, gates, pathway_updates):
        expected = {"sensory", "value", "map", "action"}
        missing = sorted(expected - set(pathway_updates))
        extra = sorted(set(pathway_updates) - expected)
        if missing or extra:
            raise ValueError(f"Invalid pathway updates: missing={missing}, extra={extra}")
        if gates.ndim != 2 or gates.shape[1] != 4:
            raise ValueError(f"Expected gates [B,4], got {tuple(gates.shape)}")
        batch = z_before.shape[0]
        if gates.shape[0] != batch:
            raise ValueError(f"Gate batch {gates.shape[0]} != latent batch {batch}")
        for name, update in pathway_updates.items():
            if update.shape != z_before.shape:
                raise ValueError(
                    f"{name} update shape {tuple(update.shape)} != z_before {tuple(z_before.shape)}")
        g = gates.view(batch, 4, 1, 1)
        z_update = (
            g[:, 0:1] * pathway_updates["sensory"]
            + g[:, 1:2] * pathway_updates["value"]
            + g[:, 2:3] * pathway_updates["map"]
            + g[:, 3:4] * pathway_updates["action"]
        )
        z_updated = z_before + z_update
        routing_gate_logits = _safe_logit(gates)
        remap_logits = self.remap_classifier(z_updated.mean(dim=(2, 3)))
        return {
            **self._decode(z_updated),
            "remap_logits": remap_logits,
            "routing_gate_logits": routing_gate_logits,
            "routing_gates": gates,
            "gates": gates,
            "gate_logits": routing_gate_logits,
            "z_before": z_before,
            "z_update": z_update,
            "z_updated": z_updated,
            "pathway_updates": pathway_updates,
        }

    def forward(self, x, gate_override=None, target_multihot=None):
        z_before = self.encoder(x[:, :N_GRID])
        z_after = self.encoder(x[:, N_GRID:])
        diff = z_after - z_before
        error_features = self.error_net(torch.cat([z_before, z_after, diff, diff.abs()], dim=1))
        routing_gate_logits = self.gate_net(error_features.mean(dim=(2, 3)))
        raw_routing_gates = torch.sigmoid(routing_gate_logits)
        routing_gates = (
            self._apply_override(raw_routing_gates, gate_override, target_multihot)
            if gate_override is not None else raw_routing_gates)
        pathway_updates = {
            "sensory": self.sensory_head(error_features),
            "value": self.value_head(error_features),
            "map": self.map_head(error_features),
            "action": self.action_head(error_features),
        }
        decoded = self.decode_from_components(z_before, routing_gates, pathway_updates)
        return {
            **decoded,
            "routing_gate_logits": routing_gate_logits,
            "routing_gates": routing_gates,
            "raw_gate_logits": routing_gate_logits,
            "raw_gates": raw_routing_gates,
            "gates": routing_gates,
            "gate_logits": routing_gate_logits,
            "z_before": z_before,
            "z_after": z_after,
            "pathway_updates": pathway_updates,
        }
