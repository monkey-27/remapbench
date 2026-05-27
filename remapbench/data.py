"""PyTorch dataset wrapper for RemapBench npz files."""
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object  # dummy base so class definition doesn't crash


class RemapDataset(Dataset):
    """
    Load a RemapBench split (.npz) as a PyTorch Dataset.

    Each __getitem__ returns a dict:
        x:              float32 [2*C, H, W]  concat(before_grid, after_grid)
        before_grid:    float32 [C, H, W]
        after_grid:     float32 [C, H, W]
        delta_future:   float32 [1, H, W]
        delta_value:    float32 [1, H, W]
        action_delta:   float32 [4, H, W]
        target_multihot:float32 [4]
        scalar_errors:  float32 [5]  nuisance/full_visual/future/value/action
        intervention_id:int64  scalar
        layout_id:      int64  scalar
        sample_id:      int64  scalar
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
        before = torch.from_numpy(d["before_grid"][idx].astype(np.float32))  # [C,H,W]
        after  = torch.from_numpy(d["after_grid"][idx].astype(np.float32))   # [C,H,W]

        ne  = float(d["nuisance_error"][idx])   if "nuisance_error"   in d else float(d.get("sensory_error", np.zeros(self._n))[idx])
        fve = float(d["full_visual_error"][idx]) if "full_visual_error" in d else 0.0
        fe  = float(d["future_error"][idx])
        ve  = float(d["value_error"][idx])
        ae  = float(d["action_error"][idx])

        scalar_errors = torch.tensor([ne, fve, fe, ve, ae], dtype=torch.float32)

        return {
            "x":               torch.cat([before, after], dim=0),
            "before_grid":     before,
            "after_grid":      after,
            "delta_future":    torch.from_numpy(d["delta_future"][idx][np.newaxis].astype(np.float32)),
            "delta_value":     torch.from_numpy(d["delta_value"][idx][np.newaxis].astype(np.float32)),
            "action_delta":    torch.from_numpy(d["action_delta"][idx].astype(np.float32)),
            "target_multihot": torch.from_numpy(d["target_multihot"][idx].astype(np.float32)),
            "scalar_errors":   scalar_errors,
            "intervention_id": torch.tensor(int(d["intervention_id"][idx]), dtype=torch.int64),
            "layout_id":       torch.tensor(int(d["layout_id"][idx]) if "layout_id" in d else idx, dtype=torch.int64),
            "sample_id":       torch.tensor(int(d["sample_id"][idx]) if "sample_id" in d else idx, dtype=torch.int64),
        }
