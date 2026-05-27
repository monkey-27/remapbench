"""
GatedERPM: Error-Gated Predictive Map with explicit latent pathway routing.

Mechanism:
  1. Encode before/after grids into shared latent spatial states.
  2. Compute error features from state differences.
  3. Pool error features → learned gate logits → sigmoid gates [B,4].
  4. Four pathway heads compute pathway-specific latent updates from error features.
  5. Gated combination: z_update = sum_k(gate_k * u_k), z_updated = z_before + z_update.
  6. Decode predictions from updated latent.

The gates are simultaneously the remap_logits (trained with BCE against target_multihot),
enforcing a tight coupling between the gating mechanism and the plasticity labels.

Gate override note:
  When gate_override is None (normal training/eval):
    - raw_gate_logits = gate_net output
    - gates = sigmoid(raw_gate_logits)
    - remap_logits = raw_gate_logits   ← used for BCE training
  When gate_override is set (ablation):
    - final_gates reflect the override
    - remap_logits = _safe_logit(final_gates)   ← ablated predictions
    - raw_gate_logits still returned for reference
"""
import torch
import torch.nn as nn

N_GRID = 10
LATENT = 64


def _safe_logit(p, eps=1e-6):
    """Numerically safe logit: clamps p away from 0/1 before log-odds."""
    p = p.clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))


class _ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))


class _GridEncoder(nn.Module):
    """Shared spatial encoder for before/after grids."""
    def __init__(self, in_ch=N_GRID, out_ch=LATENT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            _ResBlock(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _PathwayHead(nn.Module):
    """Pathway-specific latent update: error_features → latent delta."""
    def __init__(self, ch=LATENT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class GatedERPM(nn.Module):
    """
    Error-Gated Predictive Map (GatedERPM).

    Forward args:
        x:                [B, 2*C, H, W]  concatenated before/after grids
        gate_override:    str or None — ablation mode (see _apply_override)
        target_multihot:  [B, 4] float tensor — required for 'oracle' override

    Returns a dict with keys:
        future_after, value_after, delta_future, delta_value, action_delta,
        remap_logits   — logits for remap prediction (raw during training, safe_logit(final) during ablation)
        raw_gate_logits — original gate_net output (always)
        raw_gates       — sigmoid(raw_gate_logits)
        gates           — final gates after any override (equals raw_gates when no override)
        gate_logits     — _safe_logit(gates)  (equals raw_gate_logits when no override)
        z_before, z_after, z_updated,
        pathway_updates (dict: sensory/value/map/action each [B,L,H,W])
    """

    def __init__(self, n_grid=N_GRID, latent=LATENT):
        super().__init__()
        L = latent
        self.encoder = _GridEncoder(n_grid, L)

        # Error feature net: concat [z_b, z_a, z_a-z_b, |z_a-z_b|] → 4L channels
        self.error_net = nn.Sequential(
            nn.Conv2d(4 * L, 2 * L, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(2 * L, L, 3, padding=1), nn.ReLU(inplace=True),
        )

        # Gate net: global avg pool → MLP → 4 gate logits
        self.gate_net = nn.Sequential(
            nn.Linear(L, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 4),
        )

        # Pathway-specific update heads (4 independent pathways)
        self.sensory_head = _PathwayHead(L)
        self.value_head   = _PathwayHead(L)
        self.map_head     = _PathwayHead(L)
        self.action_head  = _PathwayHead(L)

        # Decoders from updated latent
        self.future_after_head = nn.Conv2d(L, 1, 1)
        self.value_after_head  = nn.Conv2d(L, 1, 1)
        self.delta_future_head = nn.Conv2d(L, 1, 1)
        self.delta_value_head  = nn.Conv2d(L, 1, 1)
        self.action_delta_head = nn.Conv2d(L, 4, 1)

    def forward(self, x, gate_override=None, target_multihot=None):
        B = x.size(0)
        z_before = self.encoder(x[:, :N_GRID])   # [B, L, H, W]
        z_after  = self.encoder(x[:, N_GRID:])   # [B, L, H, W]

        diff = z_after - z_before
        err_in = torch.cat([z_before, z_after, diff, diff.abs()], dim=1)
        error_features = self.error_net(err_in)   # [B, L, H, W]

        pooled = error_features.mean(dim=(2, 3))  # [B, L]
        raw_gate_logits = self.gate_net(pooled)   # [B, 4]
        raw_gates = torch.sigmoid(raw_gate_logits)

        if gate_override is not None:
            final_gates = self._apply_override(raw_gates, gate_override, target_multihot)
            final_gate_logits = _safe_logit(final_gates)
            remap_logits = final_gate_logits
        else:
            final_gates = raw_gates
            final_gate_logits = raw_gate_logits
            remap_logits = raw_gate_logits

        u_sensory = self.sensory_head(error_features)
        u_value   = self.value_head(error_features)
        u_map     = self.map_head(error_features)
        u_action  = self.action_head(error_features)

        g = final_gates.view(B, 4, 1, 1)
        z_update  = (g[:, 0:1] * u_sensory + g[:, 1:2] * u_value
                     + g[:, 2:3] * u_map + g[:, 3:4] * u_action)
        z_updated = z_before + z_update

        return {
            "future_after":     self.future_after_head(z_updated),
            "value_after":      self.value_after_head(z_updated),
            "delta_future":     self.delta_future_head(z_updated),
            "delta_value":      self.delta_value_head(z_updated),
            "action_delta":     self.action_delta_head(z_updated),
            "remap_logits":     remap_logits,
            "raw_gate_logits":  raw_gate_logits,
            "raw_gates":        raw_gates,
            "gates":            final_gates,
            "gate_logits":      final_gate_logits,
            "z_before":         z_before,
            "z_after":          z_after,
            "z_updated":        z_updated,
            "pathway_updates": {
                "sensory": u_sensory,
                "value":   u_value,
                "map":     u_map,
                "action":  u_action,
            },
        }

    def _apply_override(self, raw_gates, mode, target_multihot):
        g = raw_gates.clone()
        if mode == "zero_sensory":  g[:, 0] = 0.0
        elif mode == "zero_value":  g[:, 1] = 0.0
        elif mode == "zero_map":    g[:, 2] = 0.0
        elif mode == "zero_action": g[:, 3] = 0.0
        elif mode == "all_zero":    g = torch.zeros_like(g)
        elif mode == "all_one":     g = torch.ones_like(g)
        elif mode == "oracle":
            if target_multihot is not None:
                g = target_multihot.float().to(raw_gates.device)
        return g
