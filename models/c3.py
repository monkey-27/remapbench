"""Counterfactual Cause Composer with exhaustive subset-search inference."""
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


CAUSE_NAMES = ("sensory", "value", "map", "action")
N_GRID_CHANNELS = 10


def enumerate_cause_subsets(device=None):
    """Return the fixed 16-way cause-subset table in binary index order."""
    rows = list(itertools.product((0.0, 1.0), repeat=len(CAUSE_NAMES)))
    return torch.tensor(rows, dtype=torch.float32, device=device)


def subset_ids(target_multihot):
    bits = (target_multihot > 0.5).long()
    weights = torch.tensor([8, 4, 2, 1], device=bits.device)
    return (bits * weights).sum(dim=1)


class CounterfactualCauseComposer(nn.Module):
    """Compose independent latent operators and choose the lowest-energy subset."""

    def __init__(self, scoring_variant="full_reconstruction", latent_channels=32,
                 sparsity_weight=0.001, energy_scale=100.0):
        super().__init__()
        if scoring_variant not in {"full_reconstruction", "delta_space", "factor_scored"}:
            raise ValueError(f"Unknown C3 scoring variant: {scoring_variant}")
        self.scoring_variant = scoring_variant
        self.sparsity_weight = sparsity_weight
        self.energy_scale = energy_scale
        self.register_buffer("cause_subsets", enumerate_cause_subsets())
        self.encoder = nn.Sequential(
            nn.Conv2d(N_GRID_CHANNELS, latent_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
            nn.ReLU(),
        )
        self.operators = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
            )
            for _ in CAUSE_NAMES
        ])
        self.grid_decoder = nn.Conv2d(latent_channels, N_GRID_CHANNELS, 1)
        self.future_decoder = nn.Conv2d(latent_channels, 1, 1)
        self.value_decoder = nn.Conv2d(latent_channels, 1, 1)
        self.action_decoder = nn.Conv2d(latent_channels, 4, 1)

    @staticmethod
    def _candidate_decode(decoder, candidates):
        batch, subsets, channels, height, width = candidates.shape
        decoded = decoder(candidates.reshape(batch * subsets, channels, height, width))
        return decoded.reshape(batch, subsets, *decoded.shape[1:])

    @staticmethod
    def _gather(candidates, indices):
        view_shape = [len(indices), 1] + [1] * (candidates.ndim - 2)
        index = indices.view(*view_shape).expand(-1, 1, *candidates.shape[2:])
        return candidates.gather(1, index).squeeze(1)

    def _compose(self, z_before):
        operator_updates = torch.stack([operator(z_before) for operator in self.operators], dim=1)
        candidate_deltas = torch.einsum("sk,bklhw->bslhw", self.cause_subsets, operator_updates)
        return operator_updates, candidate_deltas, z_before.unsqueeze(1) + candidate_deltas

    def forward(self, before_grid, after_grid, factor_targets=None):
        z_before = self.encoder(before_grid)
        z_after = self.encoder(after_grid)
        operator_updates, candidate_deltas, candidates = self._compose(z_before)
        candidate_grids = self._candidate_decode(self.grid_decoder, candidates)
        candidate_future = self._candidate_decode(self.future_decoder, candidates)
        candidate_value = self._candidate_decode(self.value_decoder, candidates)
        candidate_action = self._candidate_decode(self.action_decoder, candidates)

        if self.scoring_variant == "full_reconstruction":
            scores = (candidate_grids - after_grid.unsqueeze(1)).square().mean(dim=(2, 3, 4))
        elif self.scoring_variant == "delta_space":
            observed_delta = z_after - z_before
            scores = (candidate_deltas - observed_delta.unsqueeze(1)).square().mean(dim=(2, 3, 4))
        else:
            if factor_targets is None:
                raise ValueError("factor_scored C3 requires factor_targets")
            scores = (candidate_grids - after_grid.unsqueeze(1)).square().mean(dim=(2, 3, 4))
            scores = scores + (
                candidate_future - factor_targets["delta_future"].unsqueeze(1)
            ).square().mean(dim=(2, 3, 4))
            scores = scores + (
                candidate_value - factor_targets["delta_value"].unsqueeze(1)
            ).square().mean(dim=(2, 3, 4))
            scores = scores + (
                candidate_action - factor_targets["action_delta"].unsqueeze(1)
            ).square().mean(dim=(2, 3, 4))
        scores = scores + self.sparsity_weight * self.cause_subsets.sum(dim=1).unsqueeze(0)
        predicted_ids = scores.argmin(dim=1)
        return {
            "scores": scores,
            "predicted_ids": predicted_ids,
            "predicted_multihot": self.cause_subsets[predicted_ids],
            "operator_updates": operator_updates,
            "candidate_deltas": candidate_deltas,
            "candidate_grids": candidate_grids,
            "candidate_future": candidate_future,
            "candidate_value": candidate_value,
            "candidate_action": candidate_action,
            "z_before": z_before,
            "z_after": z_after,
        }

    def loss(self, batch):
        factor_targets = {key: batch[key] for key in ("delta_future", "delta_value", "action_delta")}
        out = self(batch["before_grid"], batch["after_grid"], factor_targets)
        true_ids = subset_ids(batch["target_multihot"])
        true_delta = self._gather(out["candidate_deltas"], true_ids)
        true_grid = self._gather(out["candidate_grids"], true_ids)
        true_future = self._gather(out["candidate_future"], true_ids)
        true_value = self._gather(out["candidate_value"], true_ids)
        true_action = self._gather(out["candidate_action"], true_ids)
        latent_target = out["z_after"] - out["z_before"]
        losses = {
            "subset_ce": F.cross_entropy(-self.energy_scale * out["scores"], true_ids),
            "latent_reconstruction": F.mse_loss(true_delta, latent_target),
            "grid_reconstruction": F.mse_loss(true_grid, batch["after_grid"]),
            "future_reconstruction": F.mse_loss(true_future, batch["delta_future"]),
            "value_reconstruction": F.mse_loss(true_value, batch["delta_value"]),
            "action_reconstruction": F.mse_loss(true_action, batch["action_delta"]),
        }
        losses["total"] = sum(losses.values())
        return losses, out


def build_c3(name, **kwargs):
    prefix = "c3_"
    if not name.startswith(prefix):
        raise ValueError(name)
    return CounterfactualCauseComposer(scoring_variant=name[len(prefix):], **kwargs)
