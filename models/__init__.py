from .erpm import ErrorGatedPredictiveMapCNN, StandardPredictiveMapCNN
from .gated_erpm import GatedERPM, GlobalPlasticity, UngatedLatent
from .cpo import CPO, FactorizedGates
from .fepo import FEPO
from .baselines import HeuristicBaseline, ALL_BASELINES

__all__ = [
    "ErrorGatedPredictiveMapCNN",
    "StandardPredictiveMapCNN",
    "GatedERPM",
    "UngatedLatent",
    "GlobalPlasticity",
    "CPO",
    "FactorizedGates",
    "FEPO",
    "HeuristicBaseline",
    "ALL_BASELINES",
]


def build_model(name: str, **kwargs):
    """Build a model by name."""
    if name in ("erpm", "ErrorGatedPredictiveMapCNN"):
        return ErrorGatedPredictiveMapCNN(**kwargs)
    elif name in ("standard", "standard_cnn", "StandardPredictiveMapCNN"):
        return StandardPredictiveMapCNN(**kwargs)
    elif name in ("gated_erpm", "GatedERPM"):
        return GatedERPM(**kwargs)
    elif name in ("ungated_latent", "UngatedLatent"):
        return UngatedLatent(**kwargs)
    elif name in ("global_plasticity", "GlobalPlasticity"):
        return GlobalPlasticity(**kwargs)
    elif name in ("cpo", "cpo_no_comp", "CPO"):
        return CPO(**kwargs)
    elif name in ("factorized_gates", "FactorizedGates"):
        return FactorizedGates(**kwargs)
    elif name in (
        "fepo",
        "fepo_no_evidence_invariance",
        "fepo_no_operator_reuse",
        "FEPO",
    ):
        return FEPO(**kwargs)
    else:
        raise ValueError(
            f"Unknown model: {name!r}. Choose from: erpm, standard_cnn, "
            "gated_erpm, ungated_latent, global_plasticity, cpo, cpo_no_comp, "
            "factorized_gates, fepo, fepo_no_evidence_invariance, "
            "fepo_no_operator_reuse")
