"""
Heuristic baselines for RemapBench evaluation.
Each baseline maps scalar_errors → predicted target_multihot (numpy, no torch).

Scalar error ordering (matches RemapDataset):
    [0] nuisance_error  (= sensory_error)
    [1] full_visual_error
    [2] future_error
    [3] value_error
    [4] action_error

Target label ordering: [sensory_update, value_remap, map_remap, action_remap]
"""
import numpy as np

TARGET_NAMES = ["sensory_update", "value_remap", "map_remap", "action_remap"]

# Default thresholds (can be overridden per baseline instance)
DEFAULT_THRESHOLDS = {
    "nuisance": 0.01,
    "full_visual": 0.01,
    "future": 0.01,
    "value": 0.01,
    "action": 0.01,
}


class HeuristicBaseline:
    """
    Base class. Subclasses implement _predict_one(scalar_errors, thresholds).
    scalar_errors: numpy array [5] = [nuisance, full_visual, future, value, action]
    Returns: numpy bool array [4] = predicted multihot
    """
    name = "base"

    def __init__(self, thresholds=None):
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    def predict(self, scalar_errors_batch):
        """scalar_errors_batch: [N, 5] → predictions [N, 4] float32"""
        preds = np.stack([
            self._predict_one(scalar_errors_batch[i]).astype(np.float32)
            for i in range(len(scalar_errors_batch))
        ])
        return preds

    def _predict_one(self, se, t=None):
        raise NotImplementedError


class SensoryGatedBaseline(HeuristicBaseline):
    """
    Predicts sensory_update=1 if nuisance_error > threshold.
    No structural remap labels.
    """
    name = "sensory_gated"

    def _predict_one(self, se, t=None):
        t = self.thresholds
        pred = np.zeros(4, dtype=bool)
        pred[0] = se[0] > t["nuisance"]   # sensory_update
        return pred


class ValueGatedBaseline(HeuristicBaseline):
    """Predicts value_remap=1 if value_error > threshold."""
    name = "value_gated"

    def _predict_one(self, se, t=None):
        t = self.thresholds
        pred = np.zeros(4, dtype=bool)
        pred[1] = se[3] > t["value"]
        return pred


class MapGatedBaseline(HeuristicBaseline):
    """Predicts map_remap=1 if future_error > threshold."""
    name = "map_gated"

    def _predict_one(self, se, t=None):
        t = self.thresholds
        pred = np.zeros(4, dtype=bool)
        pred[2] = se[2] > t["future"]
        return pred


class ActionGatedBaseline(HeuristicBaseline):
    """Predicts action_remap=1 if action_error > threshold."""
    name = "action_gated"

    def _predict_one(self, se, t=None):
        t = self.thresholds
        pred = np.zeros(4, dtype=bool)
        pred[3] = se[4] > t["action"]
        return pred


class GlobalPlasticityBaseline(HeuristicBaseline):
    """
    If any scalar error exceeds its threshold, predicts all active pathways
    based on which errors are above threshold. Broadly activates all channels
    when any signal is detected.
    """
    name = "global_plasticity"

    def _predict_one(self, se, t=None):
        t = self.thresholds
        pred = np.zeros(4, dtype=bool)
        if se[0] > t["nuisance"] or se[1] > t["full_visual"]:
            pred[0] = True   # sensory_update
        if se[3] > t["value"]:
            pred[1] = True   # value_remap
        if se[2] > t["future"]:
            pred[2] = True   # map_remap
        if se[4] > t["action"]:
            pred[3] = True   # action_remap
        # If nothing activated, predict all four (worst-case overfiring)
        if not pred.any():
            pred[:] = True
        return pred


class OracleBaseline(HeuristicBaseline):
    """Returns the ground-truth target_multihot (perfect predictions)."""
    name = "oracle"

    def predict(self, scalar_errors_batch, targets=None):
        if targets is None:
            raise ValueError("OracleBaseline.predict requires targets array")
        return targets.astype(np.float32)

    def _predict_one(self, se, t=None):
        raise NotImplementedError("Use predict() with targets kwarg")


ALL_BASELINES = [
    SensoryGatedBaseline,
    ValueGatedBaseline,
    MapGatedBaseline,
    ActionGatedBaseline,
    GlobalPlasticityBaseline,
    OracleBaseline,
]
