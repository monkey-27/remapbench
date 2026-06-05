"""PyTorch dataset wrapper for RemapBench npz files."""
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object


class RemapDataset(Dataset):
    """
    Load a RemapBench split (.npz) as a PyTorch Dataset.

    Each __getitem__ returns a dict:
        x:              float32 [2*C, H, W]  concat(before_grid, after_grid)
        before_grid:    float32 [C, H, W]
        after_grid:     float32 [C, H, W]
        future_before:  float32 [1, H, W]
        future_after:   float32 [1, H, W]
        value_before:   float32 [1, H, W]
        value_after:    float32 [1, H, W]
        delta_future:   float32 [1, H, W]
        delta_value:    float32 [1, H, W]
        action_delta:   float32 [4, H, W]
        target_multihot:float32 [4]
        scalar_errors:  float32 [5]  nuisance/full_visual/future/value/action
        intervention_id:int64  scalar
        layout_id:      int64  scalar
        sample_id:      int64  scalar
        start_xy:       int64  [2]
        goal_after_xy:  int64  [2]
    """

    def __init__(self, path):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for RemapDataset.")
        raw = np.load(path, allow_pickle=True)
        self._data = {k: raw[k] for k in raw.files}
        self._n = len(self._data["intervention_id"])

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        d = self._data

        def _t(key, dtype=np.float32):
            return torch.from_numpy(d[key][idx].astype(dtype))

        def _t1(key):
            """Add channel dim: [H,W] → [1,H,W]."""
            return torch.from_numpy(d[key][idx][np.newaxis].astype(np.float32))

        before = _t("before_grid")  # [C,H,W]
        after  = _t("after_grid")   # [C,H,W]

        ne  = float(d["nuisance_error"][idx])   if "nuisance_error"   in d else float(d.get("sensory_error", np.zeros(self._n))[idx])
        fve = float(d["full_visual_error"][idx]) if "full_visual_error" in d else 0.0
        fe  = float(d["future_error"][idx])
        ve  = float(d["value_error"][idx])
        ae  = float(d["action_error"][idx])

        item = {
            "x":               torch.cat([before, after], dim=0),
            "before_grid":     before,
            "after_grid":      after,
            "future_before":   _t1("future_before"),
            "future_after":    _t1("future_after"),
            "value_before":    _t1("value_before"),
            "value_after":     _t1("value_after"),
            "delta_future":    _t1("delta_future"),
            "delta_value":     _t1("delta_value"),
            "action_delta":    _t("action_delta"),
            "target_multihot": torch.from_numpy(d["target_multihot"][idx].astype(np.float32)),
            "scalar_errors":   torch.tensor([ne, fve, fe, ve, ae], dtype=torch.float32),
            "intervention_id": torch.tensor(int(d["intervention_id"][idx]), dtype=torch.int64),
            "layout_id":       torch.tensor(int(d["layout_id"][idx]) if "layout_id" in d else idx, dtype=torch.int64),
            "sample_id":       torch.tensor(int(d["sample_id"][idx]) if "sample_id" in d else idx, dtype=torch.int64),
            "start_xy":        torch.from_numpy(d["start_xy"][idx].astype(np.int64)),
            "goal_after_xy":   torch.from_numpy(d["goal_after_xy"][idx].astype(np.int64)),
        }
        if "intervention_pair_id" in d:
            item["intervention_pair_id"] = torch.tensor(int(d["intervention_pair_id"][idx]), dtype=torch.int64)
        if "tuple_id" in d:
            item.update(
                tuple_id=torch.tensor(int(d["tuple_id"][idx]), dtype=torch.int64),
                tuple_role_id=torch.tensor(int(d["tuple_role_id"][idx]), dtype=torch.int64),
                tuple_pair_id=torch.tensor(int(d["tuple_pair_id"][idx]), dtype=torch.int64),
            )
        return item


class TupleRemapDataset(Dataset):
    """Group linked base/A/B/AB rows from a tuple split into one training item."""

    ROLES = {"base": 0, "A": 1, "B": 2, "AB": 3}

    def __init__(self, path):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for TupleRemapDataset.")
        self.rows = RemapDataset(path)
        d = self.rows._data
        if "tuple_id" not in d or "tuple_role_id" not in d:
            raise ValueError(f"{path} is not a linked tuple split")
        self._groups = []
        for tuple_id in sorted(set(int(x) for x in d["tuple_id"])):
            indices = np.where(d["tuple_id"] == tuple_id)[0]
            by_role = {int(d["tuple_role_id"][idx]): int(idx) for idx in indices}
            if set(by_role) != set(self.ROLES.values()):
                raise ValueError(f"tuple_id={tuple_id} does not contain base/A/B/AB exactly once")
            self._groups.append((tuple_id, by_role))

    def __len__(self):
        return len(self._groups)

    def __getitem__(self, idx):
        tuple_id, by_role = self._groups[idx]
        return {
            "tuple_id": torch.tensor(tuple_id, dtype=torch.int64),
            "base": self.rows[by_role[0]],
            "A": self.rows[by_role[1]],
            "B": self.rows[by_role[2]],
            "AB": self.rows[by_role[3]],
        }
