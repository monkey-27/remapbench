from .erpm import ErrorGatedPredictiveMapCNN, StandardPredictiveMapCNN
from .gated_erpm import GatedERPM, GlobalPlasticity, UngatedLatent
from .decoupled_gated_erpm import DecoupledGatedERPM
from .cpo import CPO, FactorizedGates
from .fepo import FEPO
from .c3 import CounterfactualCauseComposer, build_c3
from .baselines import HeuristicBaseline, ALL_BASELINES

__all__ = [
    "ErrorGatedPredictiveMapCNN",
    "StandardPredictiveMapCNN",
    "GatedERPM",
    "DecoupledGatedERPM",
    "UngatedLatent",
    "GlobalPlasticity",
    "CPO",
    "FactorizedGates",
    "FEPO",
    "CounterfactualCauseComposer",
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
    elif name in ("decoupled_gated_erpm", "DecoupledGatedERPM"):
        return DecoupledGatedERPM(**kwargs)
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
        "fepo_directonly",
        "fepo_tuple_supervised",
        "FEPO",
    ):
        return FEPO(**kwargs)
    elif name.startswith("c3_"):
        return build_c3(name, **kwargs)
    else:
        raise ValueError(
            f"Unknown model: {name!r}. Choose from: erpm, standard_cnn, "
            "gated_erpm, decoupled_gated_erpm, ungated_latent, global_plasticity, cpo, cpo_no_comp, "
            "factorized_gates, fepo, fepo_no_evidence_invariance, "
            "fepo_no_operator_reuse, fepo_directonly, fepo_tuple_supervised, "
            "c3_full_reconstruction, c3_delta_space, c3_factor_scored")
