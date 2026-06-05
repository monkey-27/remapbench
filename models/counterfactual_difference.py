"""Compact primitive-evidence cause detector."""
import torch
import torch.nn as nn


class DiffCauseNet(nn.Module):
    """Small CNN over before/after/diff channels with per-cause evidence maps."""

    def __init__(
        self,
        in_channels=10,
        hidden_channels=48,
        use_diff_channels=True,
        label_from_evidence_pool=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.use_diff_channels = use_diff_channels
        self.label_from_evidence_pool = label_from_evidence_pool
        model_channels = in_channels * (4 if use_diff_channels else 2)
        self.encoder = nn.Sequential(
            nn.Conv2d(model_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
        )
        self.evidence_head = nn.Conv2d(hidden_channels, 4, 1)
        self.classifiers = nn.ModuleList(nn.Linear(hidden_channels, 1) for _ in range(4))

    def _input(self, x=None, before_grid=None, after_grid=None):
        if before_grid is None or after_grid is None:
            if x is None:
                raise ValueError("DiffCauseNet needs x or before_grid/after_grid")
            before_grid, after_grid = x[:, :self.in_channels], x[:, self.in_channels:]
        if not self.use_diff_channels:
            return torch.cat([before_grid, after_grid], dim=1)
        signed = after_grid - before_grid
        return torch.cat([before_grid, after_grid, signed, signed.abs()], dim=1)

    def forward(self, x=None, before_grid=None, after_grid=None):
        features = self.encoder(self._input(x, before_grid, after_grid))
        evidence_logits = self.evidence_head(features)
        evidence_maps = torch.sigmoid(evidence_logits)
        if self.label_from_evidence_pool:
            logits = evidence_logits.flatten(2).amax(dim=2)
            vectors = []
            for k in range(4):
                weights = evidence_maps[:, k:k + 1]
                denom = weights.sum(dim=(2, 3), keepdim=True).clamp_min(1e-4)
                vectors.append(((features * weights).sum(dim=(2, 3), keepdim=True) / denom).flatten(1))
            return {
                "cause_logits": logits,
                "remap_logits": logits,
                "evidence_maps": evidence_maps,
                "evidence_logits": evidence_logits,
                "evidence_vectors": torch.stack(vectors, dim=1),
            }
        vectors = []
        logits = []
        for k, classifier in enumerate(self.classifiers):
            weights = evidence_maps[:, k:k + 1]
            denom = weights.sum(dim=(2, 3), keepdim=True).clamp_min(1e-4)
            vec = (features * weights).sum(dim=(2, 3), keepdim=True) / denom
            vec = vec.flatten(1)
            vectors.append(vec)
            logits.append(classifier(vec))
        return {
            "cause_logits": torch.cat(logits, dim=1),
            "remap_logits": torch.cat(logits, dim=1),
            "evidence_maps": evidence_maps,
            "evidence_logits": evidence_logits,
            "evidence_vectors": torch.stack(vectors, dim=1),
        }


CounterfactualDifferenceModel = DiffCauseNet
EvidenceMaskCauseNet = DiffCauseNet
