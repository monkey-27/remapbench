"""Factorized evidence pathway operators for selective latent remapping."""
import torch
import torch.nn as nn

from .gated_erpm import LATENT, N_GRID, _GridEncoder, _safe_logit


PATHWAYS = ("sensory", "value", "map", "action")

class _EvidenceEncoder(nn.Module):
    """Extract one pathway-specific evidence map from shared error features."""

    def __init__(self, latent):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent, latent, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent, latent, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, evidence):
        return self.net(evidence)


class _EvidenceOperator(nn.Module):
    """Produce one latent update from z_before and one pathway's evidence."""

    def __init__(self, latent):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2 * latent, latent, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent, latent, 3, padding=1),
        )

    def forward(self, z_before, evidence_features):
        return self.net(torch.cat([z_before, evidence_features], dim=1))


class _GateHead(nn.Module):
    """Predict one pathway gate from that pathway's evidence alone."""

    def __init__(self, latent):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, evidence_features):
        return self.net(evidence_features.mean(dim=(2, 3)))


class FEPO(nn.Module):
    """Factorized Evidence Pathway Operators.

    Every operator consumes only the shared pre-change state and evidence from
    its own pathway. The three registry ablations intentionally use this exact
    architecture; their training configs determine which losses are disabled.
    """

    composition_losses_enabled = True

    def __init__(self, n_grid=N_GRID, latent=LATENT):
        super().__init__()
        self.n_grid = n_grid
        self.encoder = _GridEncoder(n_grid, latent)
        self.error_net = nn.Sequential(
            nn.Conv2d(4 * latent, 2 * latent, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * latent, latent, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.evidence_encoders = nn.ModuleDict({
            name: _EvidenceEncoder(latent) for name in PATHWAYS
        })
        self.operators = nn.ModuleDict({
            name: _EvidenceOperator(latent) for name in PATHWAYS
        })
        self.gate_heads = nn.ModuleDict({
            name: _GateHead(latent) for name in PATHWAYS
        })

        self.future_after_head = nn.Conv2d(latent, 1, 1)
        self.value_after_head = nn.Conv2d(latent, 1, 1)
        self.delta_future_head = nn.Conv2d(latent, 1, 1)
        self.delta_value_head = nn.Conv2d(latent, 1, 1)
        self.action_delta_head = nn.Conv2d(latent, 4, 1)

    def _decode(self, z_updated):
        return {
            "future_after": self.future_after_head(z_updated),
            "value_after": self.value_after_head(z_updated),
            "delta_future": self.delta_future_head(z_updated),
            "delta_value": self.delta_value_head(z_updated),
            "action_delta": self.action_delta_head(z_updated),
        }

    def forward(self, x, gate_override=None, target_multihot=None):
        before, after = x[:, :self.n_grid], x[:, self.n_grid:]
        z_before = self.encoder(before)
        z_after = self.encoder(after)
        diff = z_after - z_before
        shared_error_features = self.error_net(
            torch.cat([z_before, z_after, diff, diff.abs()], dim=1))

        evidence_features = {}
        evidence_vectors = {}
        operator_inputs = {}
        pathway_updates = {}
        gate_logits = []
        for name in PATHWAYS:
            evidence_features[name] = self.evidence_encoders[name](shared_error_features)
            evidence_vectors[name] = evidence_features[name].mean(dim=(2, 3))
            operator_inputs[name] = torch.cat([z_before, evidence_features[name]], dim=1)
            pathway_updates[name] = self.operators[name](z_before, evidence_features[name])
            gate_logits.append(self.gate_heads[name](evidence_features[name]))

        raw_gate_logits = torch.cat(gate_logits, dim=1)
        raw_gates = torch.sigmoid(raw_gate_logits)
        if gate_override is None:
            gates = raw_gates
            final_gate_logits = raw_gate_logits
        else:
            gates = self._apply_override(raw_gates, gate_override, target_multihot)
            final_gate_logits = _safe_logit(gates)

        g = gates.view(x.size(0), len(PATHWAYS), 1, 1)
        z_update = sum(
            g[:, index:index + 1] * pathway_updates[name]
            for index, name in enumerate(PATHWAYS)
        )
        z_updated = z_before + z_update
        return {
            **self._decode(z_updated),
            "remap_logits": final_gate_logits,
            "raw_gate_logits": raw_gate_logits,
            "raw_gates": raw_gates,
            "gates": gates,
            "gate_logits": final_gate_logits,
            "z_before": z_before,
            "z_after": z_after,
            "z_update": z_update,
            "z_updated": z_updated,
            "shared_error_features": shared_error_features,
            "pathway_evidence_features": evidence_features,
            "evidence_vectors": evidence_vectors,
            "operator_inputs": operator_inputs,
            "pathway_updates": pathway_updates,
        }

    def forward_tuple(self, x_left, x_right, gate_override=None, target_multihot=None):
        """Compose independently computed evidence-local pathway updates."""
        left = self.forward(x_left)
        right = self.forward(x_right)
        gates = 1.0 - (1.0 - left["gates"]) * (1.0 - right["gates"])
        if gate_override is not None:
            gates = self._apply_override(gates, gate_override, target_multihot)
        gate_logits = _safe_logit(gates)

        pathway_updates = {}
        additive_updates = {}
        for index, name in enumerate(PATHWAYS):
            left_update = left["gates"][:, index:index + 1, None, None] * left["pathway_updates"][name]
            right_update = right["gates"][:, index:index + 1, None, None] * right["pathway_updates"][name]
            additive_updates[name] = left_update + right_update
            pathway_updates[name] = additive_updates[name]

        z_update = sum(pathway_updates.values())
        z_updated = left["z_before"] + z_update
        residual_loss = z_update.sum() * 0.0
        return {
            **self._decode(z_updated),
            "remap_logits": gate_logits,
            "raw_gate_logits": gate_logits,
            "raw_gates": gates,
            "gates": gates,
            "gate_logits": gate_logits,
            "z_before": left["z_before"],
            "z_after": right["z_after"],
            "z_update": z_update,
            "z_updated": z_updated,
            "pathway_updates": pathway_updates,
            "tuple_composition_loss": residual_loss,
            "tuple_composition_diagnostics": {
                "left_gates": left["gates"],
                "right_gates": right["gates"],
                "additive_pathway_updates": additive_updates,
            },
        }

    @staticmethod
    def _apply_override(raw_gates, mode, target_multihot):
        gates = raw_gates.clone()
        if mode == "zero_sensory":
            gates[:, 0] = 0.0
        elif mode == "zero_value":
            gates[:, 1] = 0.0
        elif mode == "zero_map":
            gates[:, 2] = 0.0
        elif mode == "zero_action":
            gates[:, 3] = 0.0
        elif mode == "all_zero":
            gates = torch.zeros_like(gates)
        elif mode == "all_one":
            gates = torch.ones_like(gates)
        elif mode == "oracle" and target_multihot is not None:
            gates = target_multihot.float().to(raw_gates.device)
        return gates
