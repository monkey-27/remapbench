"""
Neural models for Error-Gated Plasticity in Predictive Neural Maps.

ErrorGatedPredictiveMapCNN (ERPM):
    Compact CNN that predicts Δfuture, Δvalue, Δaction, and remap_logits
    from the concatenated before/after grid.

StandardPredictiveMapCNN:
    Same architecture as ERPM. Intended as a baseline; can be trained with
    reduced type_weight to suppress the categorical remap head.

Both are architecture-equivalent for this pilot; future versions should
incorporate explicit error-gating mechanisms to strengthen ERPM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Input: [B, 2*C, H, W] where C = 10 grid channels → 20 input channels
N_GRID_CHANNELS = 10
IN_CHANNELS = 2 * N_GRID_CHANNELS   # 20


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))


class _BackboneHeads(nn.Module):
    """Shared backbone + four prediction heads."""

    def __init__(self, in_channels=IN_CHANNELS, base=32, spatial=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, spatial, 3, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(spatial),
            nn.Conv2d(spatial, spatial, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.future_head = nn.Conv2d(spatial, 1, 1)
        self.value_head  = nn.Conv2d(spatial, 1, 1)
        self.action_head = nn.Conv2d(spatial, 4, 1)
        self.remap_head  = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(spatial, 4),
        )

    def forward(self, x):
        feat = self.backbone(x)
        return {
            "delta_future":  self.future_head(feat),   # [B,1,H,W]
            "delta_value":   self.value_head(feat),    # [B,1,H,W]
            "action_delta":  self.action_head(feat),   # [B,4,H,W]
            "remap_logits":  self.remap_head(feat),    # [B,4]
        }


class ErrorGatedPredictiveMapCNN(_BackboneHeads):
    """
    ERPM model. Error-gated plasticity routes sensory, value, future-state, and
    action prediction errors to distinct update pathways. In this pilot the
    gating is implicit: the remap head predicts which pathway should update,
    trained with a multi-label BCE objective.
    """
    pass


class StandardPredictiveMapCNN(_BackboneHeads):
    """
    Standard predictive map baseline. Same architecture as ERPM. Can be trained
    with type_weight=0.0 (or low) to suppress the categorical remap head and
    rely only on map prediction losses.
    """
    pass


def build_model(name: str, **kwargs):
    if name in ("erpm", "ErrorGatedPredictiveMapCNN"):
        return ErrorGatedPredictiveMapCNN(**kwargs)
    elif name in ("standard", "StandardPredictiveMapCNN"):
        return StandardPredictiveMapCNN(**kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")
