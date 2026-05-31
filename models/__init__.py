from .erpm import ErrorGatedPredictiveMapCNN, StandardPredictiveMapCNN
from .gated_erpm import GatedERPM, GlobalPlasticity, UngatedLatent
from .baselines import HeuristicBaseline, ALL_BASELINES

__all__ = [
    "ErrorGatedPredictiveMapCNN",
    "StandardPredictiveMapCNN",
    "GatedERPM",
    "UngatedLatent",
    "GlobalPlasticity",
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
    else:
        raise ValueError(
            f"Unknown model: {name!r}. Choose from: erpm, standard_cnn, "
            "gated_erpm, ungated_latent, global_plasticity")
