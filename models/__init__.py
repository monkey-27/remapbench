from .erpm import ErrorGatedPredictiveMapCNN, StandardPredictiveMapCNN
from .gated_erpm import GatedERPM
from .baselines import HeuristicBaseline, ALL_BASELINES

__all__ = [
    "ErrorGatedPredictiveMapCNN",
    "StandardPredictiveMapCNN",
    "GatedERPM",
    "HeuristicBaseline",
    "ALL_BASELINES",
]


def build_model(name: str, **kwargs):
    """Build a model by name. Names: erpm, standard, gated_erpm."""
    if name in ("erpm", "ErrorGatedPredictiveMapCNN"):
        return ErrorGatedPredictiveMapCNN(**kwargs)
    elif name in ("standard", "StandardPredictiveMapCNN"):
        return StandardPredictiveMapCNN(**kwargs)
    elif name in ("gated_erpm", "GatedERPM"):
        return GatedERPM(**kwargs)
    else:
        raise ValueError(f"Unknown model: {name!r}. Choose from: erpm, standard, gated_erpm")
