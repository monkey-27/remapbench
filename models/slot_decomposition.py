"""Residual-grounded cause slots for mixed-evidence decomposition."""
import torch
import torch.nn as nn

from .gated_erpm import LATENT, N_GRID, _GridEncoder


PATHWAYS = ("sensory", "value", "map", "action")
RESIDUAL_CHANNELS = 6  # future delta, value delta, four action deltas


class SlotDecomposition(nn.Module):
    """Predict each cause from its own spatially attended residual slot."""

    def __init__(self, n_grid=N_GRID, latent=LATENT, shared_pooled=False):
        super().__init__()
        self.n_grid = n_grid
        self.latent = latent
        self.shared_pooled = shared_pooled
        self.encoder = _GridEncoder(n_grid, latent)
        self.delta_net = nn.Sequential(
            nn.Conv2d(4 * latent, 2 * latent, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(2 * latent, latent, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.key_net = nn.Conv2d(latent, latent, 1)
        self.cause_queries = nn.Parameter(torch.randn(len(PATHWAYS), latent) * 0.02)
        self.classifiers = nn.ModuleList([nn.Linear(latent, 1) for _ in PATHWAYS])
        self.routers = nn.ModuleList([nn.Linear(latent, 1) for _ in PATHWAYS])
        self.residual_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(latent, latent, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(latent, RESIDUAL_CHANNELS, 1),
            )
            for _ in PATHWAYS
        ])
        self.future_after_head = nn.Conv2d(latent + RESIDUAL_CHANNELS, 1, 1)
        self.value_after_head = nn.Conv2d(latent + RESIDUAL_CHANNELS, 1, 1)

    @staticmethod
    def residual_target(batch):
        return torch.cat([batch["delta_future"], batch["delta_value"], batch["action_delta"]], dim=1)

    def forward(self, x, routing_override=None, target_multihot=None):
        before, after = x[:, :self.n_grid], x[:, self.n_grid:]
        features_before = self.encoder(before)
        features_after = self.encoder(after)
        diff = features_after - features_before
        delta_map = self.delta_net(torch.cat(
            [features_before, features_after, diff, diff.abs()], dim=1))
        keys = self.key_net(delta_map)
        batch, _, height, width = keys.shape
        scores = torch.einsum("bchw,kc->bkhw", keys, self.cause_queries)
        if self.shared_pooled:
            attention = torch.full_like(scores, 1.0 / (height * width))
        else:
            attention = scores.flatten(2).softmax(dim=2).reshape(batch, len(PATHWAYS), height, width)
        slot_maps = attention.unsqueeze(2) * delta_map.unsqueeze(1)
        slot_vectors = slot_maps.sum(dim=(3, 4))
        remap_logits = torch.cat([
            head(slot_vectors[:, index]) for index, head in enumerate(self.classifiers)
        ], dim=1)
        routing_logits = torch.cat([
            head(slot_vectors[:, index]) for index, head in enumerate(self.routers)
        ], dim=1)
        routing_weights = torch.sigmoid(routing_logits)
        if routing_override == "oracle" and target_multihot is not None:
            routing_weights = target_multihot.float().to(x.device)
        slot_residuals = torch.stack([
            decoder(slot_maps[:, index]) for index, decoder in enumerate(self.residual_decoders)
        ], dim=1)
        residual_total = (
            routing_weights[:, :, None, None, None] * slot_residuals
        ).sum(dim=1)
        updated = torch.cat([features_before, residual_total], dim=1)
        return {
            "remap_logits": remap_logits,
            "routing_logits": routing_logits,
            "routing_weights": routing_weights,
            "gates": routing_weights,
            "slot_vectors": slot_vectors,
            "slot_attention": attention,
            "slot_residuals": slot_residuals,
            "residual_total": residual_total,
            "future_after": self.future_after_head(updated),
            "value_after": self.value_after_head(updated),
            "delta_future": residual_total[:, 0:1],
            "delta_value": residual_total[:, 1:2],
            "action_delta": residual_total[:, 2:6],
            "features_before": features_before,
            "features_after": features_after,
            "delta_map": delta_map,
        }
