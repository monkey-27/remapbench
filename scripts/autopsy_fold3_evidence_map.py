"""Fold3 autopsy for the final evidence-map noisy-or Modal results."""
import argparse
import gc
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.counterfactual_difference import DiffCauseNet
from remapbench.counterfactual_difference import (  # noqa: E402
    CAUSES,
    ROLE_TO_ID,
    CounterfactualTransitionDataset,
    primitive_masks,
)


FOLDS = [
    "fold0_goal_topology",
    "fold1_sensory_action",
    "fold2_goal_action",
    "fold3_sensory_topology",
]
CAUSE_LABELS = {
    "sensory": "sensory",
    "value": "value",
    "map": "topology",
    "action": "action",
}
MODEL_FAMILIES = {
    "evidence_map_noisy_or": ("evidence", {"label_readout": "evidence_noisy_or"}),
    "learned_classifier_mask": ("evidence", {"label_readout": "learned_classifier"}),
    "evidence_mask_learned_classifier": ("evidence", {"label_readout": "learned_classifier"}),
    "grid_token_transformer": ("transformer", {}),
    "cell_graph_gnn": ("gnn", {}),
}


def subset_key(bits):
    return "".join(str(int(x)) for x in bits)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def quantiles(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    qs = [0, 5, 25, 50, 75, 95, 100]
    vals = np.percentile(arr, qs)
    return {str(q): float(v) for q, v in zip(qs, vals)}


def per_cause_metrics(pred, target, threshold=0.5):
    pred_b = pred > threshold
    target_b = target > 0.5
    out = {}
    for i, cause in enumerate(CAUSES):
        tp = int((pred_b[:, i] & target_b[:, i]).sum())
        fp = int((pred_b[:, i] & ~target_b[:, i]).sum())
        tn = int((~pred_b[:, i] & ~target_b[:, i]).sum())
        fn = int((~pred_b[:, i] & target_b[:, i]).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        out[cause] = {
            "label": CAUSE_LABELS[cause],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": fp / max(1, fp + tn),
            "false_negative_rate": fn / max(1, fn + tp),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
    exact = float((pred_b == target_b).all(axis=1).mean())
    return {"exact": exact, "per_cause": out}


def subset_confusion(pred, target, threshold=0.5):
    pred_b = pred > threshold
    target_b = target > 0.5
    c = Counter()
    for t, p in zip(target_b, pred_b):
        c[f"{subset_key(t)}->{subset_key(p)}"] += 1
    return dict(c.most_common())


def topk_mean(maps, k=3):
    flat = maps.reshape(maps.shape[0], maps.shape[1], -1)
    k = max(1, min(int(k), flat.shape[2]))
    return np.sort(flat, axis=2)[:, :, -k:].mean(axis=2)


def max_score(maps):
    return maps.reshape(maps.shape[0], maps.shape[1], -1).max(axis=2)


class GridTokenTransformer(nn.Module):
    def __init__(self, in_channels=40, hidden=96, heads=4, layers=3, max_tokens=256):
        super().__init__()
        self.proj = nn.Linear(in_channels, hidden)
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, hidden))
        enc = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(hidden, 4)

    def forward(self, before_grid=None, after_grid=None):
        diff = after_grid - before_grid
        x = torch.cat([before_grid, after_grid, diff, diff.abs()], dim=1)
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.proj(tokens) + self.pos[:, :tokens.shape[1]]
        logits = self.head(self.enc(tokens).mean(dim=1))
        return {"cause_logits": logits, "remap_logits": logits}


class CellGraphGNN(nn.Module):
    def __init__(self, in_channels=40, hidden=96, layers=4):
        super().__init__()
        self.inp = nn.Conv2d(in_channels, hidden, 1)
        self.layers = nn.ModuleList(nn.Conv2d(hidden * 2, hidden, 1) for _ in range(layers))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(hidden, 4))

    def forward(self, before_grid=None, after_grid=None):
        diff = after_grid - before_grid
        h = self.inp(torch.cat([before_grid, after_grid, diff, diff.abs()], dim=1))
        for layer in self.layers:
            nbr = (
                torch.roll(h, 1, 2) + torch.roll(h, -1, 2)
                + torch.roll(h, 1, 3) + torch.roll(h, -1, 3)
            ) / 4.0
            h = F.relu(layer(torch.cat([h, nbr], dim=1)))
        logits = self.head(h)
        return {"cause_logits": logits, "remap_logits": logits}


def make_model(model_key):
    family, kwargs = MODEL_FAMILIES[model_key]
    if family == "evidence":
        return DiffCauseNet(readout_tau=0.5, readout_topk=3, **kwargs)
    if family == "transformer":
        return GridTokenTransformer()
    if family == "gnn":
        return CellGraphGNN()
    raise ValueError(model_key)


def checkpoint_path(results_root, model_key, fold_name, seed):
    run = f"final_evidence_map_{model_key}_{fold_name}_seed{seed}"
    path = results_root / "runs" / run / "best_val_composed_seen_exact.pt"
    if not path.exists() and model_key == "learned_classifier_mask":
        run = f"final_evidence_map_evidence_mask_learned_classifier_{fold_name}_seed{seed}"
        path = results_root / "runs" / run / "best_val_composed_seen_exact.pt"
    return path


def load_model(results_root, model_key, fold_name, seed, device):
    ckpt_path = checkpoint_path(results_root, model_key, fold_name, seed)
    ckpt = torch.load(ckpt_path, map_location=device)
    model = make_model(model_key).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def collect_model(model, split_path, device, batch_size=256, include_grids=False):
    ds = CounterfactualTransitionDataset(str(split_path))
    loader = DataLoader(ds, batch_size=batch_size)
    pred, logits, target, masks, mask_targets, roles = [], [], [], [], [], []
    pair_ids, sample_ids, layout_ids = [], [], []
    before_grids, after_grids = [], []
    for batch in loader:
        before = batch["before_grid"].to(device)
        after = batch["after_grid"].to(device)
        out = model(before_grid=before, after_grid=after)
        logit = out.get("cause_logits", out.get("remap_logits"))
        prob = torch.sigmoid(logit)
        pred.append(prob.cpu().numpy())
        logits.append(logit.cpu().numpy())
        target.append(batch["target_multihot"].numpy())
        if "evidence_maps" in out:
            masks.append(out["evidence_maps"].cpu().numpy())
        else:
            h, w = before.shape[2], before.shape[3]
            masks.append(np.zeros((before.shape[0], 4, h, w), dtype=np.float32))
        mask_targets.append(batch["primitive_masks"].numpy())
        roles.extend(batch["transition_role_id"].numpy().tolist())
        pair_ids.extend(batch.get("pair_id", torch.full_like(batch["transition_role_id"], -1)).numpy().tolist())
        sample_ids.extend(batch.get("sample_id", torch.full_like(batch["transition_role_id"], -1)).numpy().tolist())
        layout_ids.extend(batch.get("layout_id", torch.full_like(batch["transition_role_id"], -1)).numpy().tolist())
        if include_grids:
            before_grids.append(before.cpu().numpy())
            after_grids.append(after.cpu().numpy())
    result = {
        "pred": np.concatenate(pred),
        "logits": np.concatenate(logits),
        "target": np.concatenate(target),
        "masks": np.concatenate(masks),
        "mask_targets": np.concatenate(mask_targets),
        "roles": np.asarray(roles),
        "pair_ids": np.asarray(pair_ids),
        "sample_ids": np.asarray(sample_ids),
        "layout_ids": np.asarray(layout_ids),
    }
    if include_grids:
        result["before_grid"] = np.concatenate(before_grids)
        result["after_grid"] = np.concatenate(after_grids)
    return result


def collect_primitive(split_path):
    ds = CounterfactualTransitionDataset(str(split_path))
    preds, targets, masks, roles, pair_ids = [], [], [], [], []
    for item in ds:
        pm = item["primitive_masks"].numpy()
        target = item["target_multihot"].numpy()
        preds.append((pm.reshape(4, -1).sum(axis=1) > 0).astype(np.float32))
        targets.append(target)
        masks.append(pm)
        roles.append(int(item["transition_role_id"]))
        pair_ids.append(int(item.get("pair_id", torch.tensor(-1))))
    return {
        "pred": np.stack(preds),
        "target": np.stack(targets),
        "masks": np.stack(masks),
        "mask_targets": np.stack(masks),
        "roles": np.asarray(roles),
        "pair_ids": np.asarray(pair_ids),
    }


def role_metrics(values):
    out = {}
    for role, rid in ROLE_TO_ID.items():
        idx = values["roles"] == rid
        if not idx.any():
            continue
        role_values = {k: v[idx] for k, v in values.items() if isinstance(v, np.ndarray) and len(v) == len(idx)}
        pred = role_values["pred"]
        target = role_values["target"]
        masks = role_values["masks"]
        mask_targets = role_values["mask_targets"]
        dice = active_dice(masks, mask_targets, target)
        out[role] = {
            "n": int(idx.sum()),
            "exact_match": float(((pred > 0.5) == (target > 0.5)).all(axis=1).mean()),
            "per_cause_recall": {
                cause: per_cause_metrics(pred, target)["per_cause"][cause]["recall"]
                for cause in CAUSES
            },
            "subset_confusion": subset_confusion(pred, target),
            "active_only_evidence_dice": dice,
        }
    return out


def active_dice(pred_maps, true_maps, target):
    pred_b = pred_maps > 0.5
    true_b = true_maps > 0.5
    out = {}
    for i, cause in enumerate(CAUSES):
        active = target[:, i] > 0.5
        if not active.any():
            out[cause] = None
            continue
        inter = (pred_b[active, i] & true_b[active, i]).sum(axis=(1, 2))
        denom = pred_b[active, i].sum(axis=(1, 2)) + true_b[active, i].sum(axis=(1, 2))
        out[cause] = float(((2 * inter + 1e-5) / (denom + 1e-5)).mean())
    return out


def mask_label_faithfulness(values):
    out = {}
    target_b = values["target"] > 0.5
    pred_b = values["pred"] > 0.5
    label_good = (pred_b == target_b).all(axis=1)
    for score_name, scorer in (("max", max_score), ("top3_mean", topk_mean)):
        scores = scorer(values["masks"])
        out[score_name] = {}
        for threshold in (0.3, 0.5, 0.7):
            mask_good = ((scores > threshold) == target_b).all(axis=1)
            out[score_name][str(threshold)] = {
                "mask_good_label_good": int((mask_good & label_good).sum()),
                "mask_good_label_bad": int((mask_good & ~label_good).sum()),
                "mask_bad_label_good": int((~mask_good & label_good).sum()),
                "mask_bad_label_bad": int((~mask_good & ~label_good).sum()),
                "n": int(len(label_good)),
            }
    return out


def active_mask_quality(values):
    pred_maps = values["masks"]
    true_maps = values["mask_targets"]
    target = values["target"] > 0.5
    pred_b = pred_maps > 0.5
    true_b = true_maps > 0.5
    max_scores = max_score(pred_maps)
    top3_scores = topk_mean(pred_maps)
    out = {}
    for i, cause in enumerate(CAUSES):
        active = target[:, i]
        inactive = ~active
        row = {"label": CAUSE_LABELS[cause]}
        if active.any():
            inter = (pred_b[active, i] & true_b[active, i]).sum(axis=(1, 2))
            union = (pred_b[active, i] | true_b[active, i]).sum(axis=(1, 2))
            denom = pred_b[active, i].sum(axis=(1, 2)) + true_b[active, i].sum(axis=(1, 2))
            row.update({
                "active_only_dice": float(((2 * inter + 1e-5) / (denom + 1e-5)).mean()),
                "active_only_iou": float(((inter + 1e-5) / (union + 1e-5)).mean()),
                "active_evidence_max": float(max_scores[active, i].mean()),
                "active_evidence_top3_mean": float(top3_scores[active, i].mean()),
                "active_evidence_area_above_0p5": float(pred_b[active, i].sum(axis=(1, 2)).mean()),
            })
        if inactive.any():
            row.update({
                "inactive_evidence_max": float(max_scores[inactive, i].mean()),
                "inactive_evidence_top3_mean": float(top3_scores[inactive, i].mean()),
                "inactive_evidence_area_above_0p5": float(pred_b[inactive, i].sum(axis=(1, 2)).mean()),
            })
        out[cause] = row
    return out


def threshold_probability_audit(values, per_seed_values, results_root, fold_name):
    print("threshold audit from Modal eval JSON", flush=True)
    per_seed = {}
    exact_fixed = []
    exact_val_exact = []
    exact_val_f1 = []
    exact_heldout = []
    for seed, seed_values in per_seed_values.items():
        run = f"final_evidence_map_evidence_map_noisy_or_{fold_name}_seed{seed}"
        path = results_root / "runs" / run / f"eval_evidence_map_noisy_or_{fold_name}_seed{seed}.json"
        report = json.loads(path.read_text())
        metrics = report["split_metrics"]["test_composed_heldout"]
        thresholds = report["threshold_calibration"]
        val_exact = thresholds["val_exact_tuned_thresholds"]
        val_f1 = thresholds["val_f1_tuned_thresholds"]
        heldout = metrics["heldout_exact_tuned_thresholds_diagnostic"]
        row = {
            "exact_fixed_0p5": exact_at_threshold(seed_values["pred"], seed_values["target"], 0.5),
            "exact_val_exact_tuned": exact_at_threshold(seed_values["pred"], seed_values["target"], val_exact),
            "exact_val_f1_tuned": exact_at_threshold(seed_values["pred"], seed_values["target"], val_f1),
            "exact_heldout_exact_tuned_diagnostic": exact_at_threshold(seed_values["pred"], seed_values["target"], heldout),
            "val_exact_tuned_thresholds": dict(zip(CAUSES, val_exact)),
            "val_f1_tuned_thresholds": dict(zip(CAUSES, val_f1)),
            "heldout_exact_tuned_thresholds_diagnostic": dict(zip(CAUSES, heldout)),
        }
        per_seed[str(seed)] = row
        exact_fixed.append(row["exact_fixed_0p5"])
        exact_val_exact.append(row["exact_val_exact_tuned"])
        exact_val_f1.append(row["exact_val_f1_tuned"])
        exact_heldout.append(row["exact_heldout_exact_tuned_diagnostic"])
    out = {
        "exact_fixed_0p5": float(np.mean(exact_fixed)),
        "exact_val_exact_tuned": float(np.mean(exact_val_exact)),
        "exact_val_f1_tuned": float(np.mean(exact_val_f1)),
        "exact_heldout_exact_tuned_diagnostic": float(np.mean(exact_heldout)),
        "per_seed": per_seed,
        "probability_quantiles": {},
    }
    for i, cause in enumerate(CAUSES):
        pos = values["target"][:, i] > 0.5
        out["probability_quantiles"][cause] = {
            "positive_samples": quantiles(values["pred"][pos, i]),
            "negative_samples": quantiles(values["pred"][~pos, i]),
        }
    return out


def exact_at_threshold(pred, target, threshold):
    if np.isscalar(threshold):
        pred_b = pred > float(threshold)
    else:
        pred_b = pred > np.asarray(threshold)[None, :]
    return float((pred_b == (target > 0.5)).all(axis=1).mean())


def tune_thresholds_f1(pred, target):
    grid = np.linspace(0.05, 0.95, 19)
    target_b = target > 0.5
    thresholds = []
    for k in range(target.shape[1]):
        best_t, best_f1 = 0.5, -1.0
        for threshold in grid:
            pred_b = pred[:, k] > threshold
            tp = int((pred_b & target_b[:, k]).sum())
            fp = int((pred_b & ~target_b[:, k]).sum())
            fn = int((~pred_b & target_b[:, k]).sum())
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-8, precision + recall)
            if f1 > best_f1:
                best_t, best_f1 = float(threshold), float(f1)
        thresholds.append(best_t)
    return thresholds


def tune_thresholds_exact_bitset(pred, target):
    grid = np.linspace(0.05, 0.95, 19)
    target_b = target > 0.5
    ok_masks = []
    for k in range(target.shape[1]):
        label_masks = []
        for threshold in grid:
            ok = (pred[:, k] > threshold) == target_b[:, k]
            bits = 0
            for i, flag in enumerate(ok):
                if flag:
                    bits |= 1 << i
            label_masks.append(bits)
        ok_masks.append(label_masks)
    best_count, best_combo = -1, [0.5] * target.shape[1]
    for a, ta in enumerate(grid):
        ma = ok_masks[0][a]
        for b, tb in enumerate(grid):
            mab = ma & ok_masks[1][b]
            for c, tc in enumerate(grid):
                mabc = mab & ok_masks[2][c]
                for d, td in enumerate(grid):
                    count = (mabc & ok_masks[3][d]).bit_count()
                    if count > best_count:
                        best_count = count
                        best_combo = [float(ta), float(tb), float(tc), float(td)]
    return best_combo


def saturation_audit(values):
    out = {}
    pred_b = values["pred"] > 0.5
    target_b = values["target"] > 0.5
    maps = values["masks"]
    for i, cause in enumerate(CAUSES):
        out[cause] = {}
        groups = {
            "true_positive_cause_maps": target_b[:, i],
            "true_negative_cause_maps": ~target_b[:, i],
            "false_positive_predictions": pred_b[:, i] & ~target_b[:, i],
            "false_negative_predictions": ~pred_b[:, i] & target_b[:, i],
        }
        for name, idx in groups.items():
            if not idx.any():
                out[cause][name] = {"n": 0}
                continue
            pix = maps[idx, i].reshape(idx.sum(), -1)
            out[cause][name] = {
                "n": int(idx.sum()),
                "mean_pixel_probability": float(pix.mean()),
                "median_pixel_probability": float(np.median(pix)),
                "max_pixel_probability": float(pix.max(axis=1).mean()),
                "pixels_gt_0p05": float((pix > 0.05).sum(axis=1).mean()),
                "pixels_gt_0p10": float((pix > 0.10).sum(axis=1).mean()),
                "pixels_gt_0p30": float((pix > 0.30).sum(axis=1).mean()),
                "pixels_gt_0p50": float((pix > 0.50).sum(axis=1).mean()),
                "noisy_or_probability": float(values["pred"][idx, i].mean()),
            }
    return out


def primitive_disagreement(split_path):
    values = collect_primitive(split_path)
    pred_b = values["pred"] > 0.5
    target_b = values["target"] > 0.5
    role_by_id = {v: k for k, v in ROLE_TO_ID.items()}
    out = {
        "primitive_diff_rule_exact": float((pred_b == target_b).all(axis=1).mean()),
        "target_active_primitive_empty": {},
        "target_inactive_primitive_nonempty": {},
        "disagreement_by_role": defaultdict(lambda: defaultdict(int)),
        "disagreement_by_pair_id": defaultdict(int),
    }
    for i, cause in enumerate(CAUSES):
        out["target_active_primitive_empty"][cause] = int((target_b[:, i] & ~pred_b[:, i]).sum())
        out["target_inactive_primitive_nonempty"][cause] = int((~target_b[:, i] & pred_b[:, i]).sum())
    bad = pred_b != target_b
    for n in range(len(pred_b)):
        if not bad[n].any():
            continue
        role = role_by_id.get(int(values["roles"][n]), str(values["roles"][n]))
        out["disagreement_by_pair_id"][str(int(values["pair_ids"][n]))] += 1
        for i, cause in enumerate(CAUSES):
            if bad[n, i]:
                out["disagreement_by_role"][role][cause] += 1
    out["disagreement_by_role"] = {k: dict(v) for k, v in out["disagreement_by_role"].items()}
    out["disagreement_by_pair_id"] = dict(Counter(out["disagreement_by_pair_id"]).most_common())
    return out


def component_stats(mask):
    mask = mask.astype(bool)
    seen = np.zeros(mask.shape, dtype=bool)
    sizes = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            size = 0
            while stack:
                rr, cc = stack.pop()
                size += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            sizes.append(size)
    return len(sizes), max(sizes) if sizes else 0


def centroid(mask):
    pts = np.argwhere(mask > 0.5)
    if len(pts) == 0:
        return None
    return pts.mean(axis=0)


def geometry_by_fold(data_root):
    out = {}
    for fold in FOLDS:
        ds = CounterfactualTransitionDataset(str(data_root / fold / "test_composed_heldout.npz"))
        per_cause = {cause: defaultdict(list) for cause in CAUSES}
        overlaps, distances = [], []
        for item in ds:
            pm = item["primitive_masks"].numpy()
            target = item["target_multihot"].numpy() > 0.5
            active_centroids = []
            for i, cause in enumerate(CAUSES):
                if not target[i]:
                    continue
                area = float((pm[i] > 0.5).sum())
                comps, max_comp = component_stats(pm[i] > 0.5)
                per_cause[cause]["primitive_mask_area"].append(area)
                per_cause[cause]["connected_component_count"].append(comps)
                per_cause[cause]["max_component_size"].append(max_comp)
                cen = centroid(pm[i])
                if cen is not None:
                    active_centroids.append(cen)
            active = np.where(target)[0]
            if len(active) >= 2:
                a, b = active[:2]
                inter = ((pm[a] > 0.5) & (pm[b] > 0.5)).sum()
                union = ((pm[a] > 0.5) | (pm[b] > 0.5)).sum()
                overlaps.append(float(inter / max(1, union)))
            if len(active_centroids) >= 2:
                distances.append(float(np.linalg.norm(active_centroids[0] - active_centroids[1])))
        out[fold] = {"causes": {}, "active_cause_mask_overlap_iou": float(np.mean(overlaps)) if overlaps else None,
                     "active_cause_centroid_distance": float(np.mean(distances)) if distances else None}
        for cause, stats in per_cause.items():
            out[fold]["causes"][cause] = {
                key: float(np.mean(vals)) if vals else None
                for key, vals in stats.items()
            }
        out[fold]["topology_edit_area"] = out[fold]["causes"]["map"].get("primitive_mask_area")
        out[fold]["sensory_mask_area"] = out[fold]["causes"]["sensory"].get("primitive_mask_area")
    return out


def make_visualizations(values, out_dir, max_each=10):
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        return {"status": "skipped", "reason": str(exc)}
    pred_b = values["pred"] > 0.5
    target_b = values["target"] > 0.5
    correct = np.where((pred_b == target_b).all(axis=1))[0][:max_each]
    common = Counter()
    for i in np.where(~(pred_b == target_b).all(axis=1))[0]:
        common[f"{subset_key(target_b[i])}->{subset_key(pred_b[i])}"] += 1
    failure = common.most_common(1)[0][0] if common else None
    incorrect = []
    for i in np.where(~(pred_b == target_b).all(axis=1))[0]:
        if failure is None or f"{subset_key(target_b[i])}->{subset_key(pred_b[i])}" == failure:
            incorrect.append(i)
        if len(incorrect) >= max_each:
            break
    index = {"common_failure": failure, "examples": []}
    for group, indices in (("correct", correct), ("incorrect", incorrect)):
        for idx in indices:
            before = values["before_grid"][idx]
            after = values["after_grid"][idx]
            changed = np.abs(after - before).sum(axis=0)
            panels = [
                ("before occupancy", before.sum(axis=0)),
                ("after occupancy", after.sum(axis=0)),
                ("changed channels", changed),
                ("primitive sensory", values["mask_targets"][idx, 0]),
                ("pred sensory", values["masks"][idx, 0]),
                ("primitive topology", values["mask_targets"][idx, 2]),
                ("pred topology", values["masks"][idx, 2]),
                ("pred value", values["masks"][idx, 1]),
                ("pred action", values["masks"][idx, 3]),
                ("primitive all", values["mask_targets"][idx].sum(axis=0)),
            ]
            tile = 92
            label_h = 28
            header_h = 52
            canvas = Image.new("RGB", (5 * tile, header_h + 2 * (tile + label_h)), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (4, 4),
                f"{group} idx={idx} target={subset_key(target_b[idx])} pred={subset_key(pred_b[idx])}",
                fill=(0, 0, 0),
            )
            draw.text((4, 22), f"prob={np.round(values['pred'][idx], 3).tolist()}", fill=(0, 0, 0))
            for panel_i, (title, img) in enumerate(panels):
                r, c = divmod(panel_i, 5)
                x0 = c * tile
                y0 = header_h + r * (tile + label_h)
                draw.text((x0 + 2, y0), title[:18], fill=(0, 0, 0))
                arr = np.asarray(img, dtype=np.float32)
                if arr.size == 0:
                    arr = np.zeros((10, 10), dtype=np.float32)
                lo, hi = float(arr.min()), float(arr.max())
                if hi > lo:
                    arr = (arr - lo) / (hi - lo)
                else:
                    arr = np.zeros_like(arr)
                rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
                rgb[..., 0] = np.clip(arr * 255, 0, 255).astype(np.uint8)
                rgb[..., 1] = np.clip(np.sqrt(arr) * 220, 0, 255).astype(np.uint8)
                rgb[..., 2] = np.clip((1 - arr) * 80, 0, 255).astype(np.uint8)
                tile_img = Image.fromarray(rgb, mode="RGB").resize((tile, tile), resample=Image.Resampling.NEAREST)
                canvas.paste(tile_img, (x0, y0 + label_h))
            path = out_dir / f"{group}_{idx}_target_{subset_key(target_b[idx])}_pred_{subset_key(pred_b[idx])}.png"
            canvas.save(path)
            index["examples"].append({
                "group": group,
                "index": int(idx),
                "path": str(path),
                "target_subset": subset_key(target_b[idx]),
                "predicted_subset": subset_key(pred_b[idx]),
                "probabilities": {cause: float(values["pred"][idx, i]) for i, cause in enumerate(CAUSES)},
            })
    return index


def summarize_report(out_dir, summary):
    noisy = summary["main_noisy"]
    conf = summary["subset_confusion"]["aggregate"]
    top_conf = list(conf.items())[:6]
    per = summary["per_cause"]["evidence_map_noisy_or"]["per_cause"]
    faith = summary["faithfulness"]["top3_mean"]["0.5"]
    sat = summary["saturation"]
    lines = [
        "# Fold3 Evidence-Map Autopsy",
        "",
        "## Main Failure Pattern",
        f"Fold3 heldout pair is sensory + topology (`1010`). Exact@0.5 for noisy-or is {noisy['exact_fixed_0p5']:.3f}.",
        "Top subset confusions: " + ", ".join(f"`{k}`={v}" for k, v in top_conf) + ".",
        "",
        "## Per-Cause Diagnosis",
    ]
    for cause in CAUSES:
        row = per[cause]
        lines.append(
            f"- {CAUSE_LABELS[cause]}: precision={row['precision']:.3f}, recall={row['recall']:.3f}, "
            f"F1={row['f1']:.3f}, FN={row['fn']}, FP={row['fp']}."
        )
    lines += [
        "",
        "## Mask vs Label Faithfulness",
        f"At top3 evidence threshold 0.5: {faith}.",
        "",
        "## Noisy-Or Saturation",
    ]
    for cause in CAUSES:
        fn = sat[cause].get("false_negative_predictions", {})
        fp = sat[cause].get("false_positive_predictions", {})
        lines.append(
            f"- {CAUSE_LABELS[cause]} FN n={fn.get('n', 0)}, mean pixels>0.05={fn.get('pixels_gt_0p05')}; "
            f"FP n={fp.get('n', 0)}, mean pixels>0.05={fp.get('pixels_gt_0p05')}."
        )
    lines += [
        "",
        "## Primitive-Diff Disagreement",
        f"Primitive diff exact on fold3 composed heldout: {summary['per_cause']['primitive_diff_rule']['exact']:.3f}.",
        f"Primitive diff exact on fold3 tuple transitions: {summary['primitive_disagreement']['primitive_diff_rule_exact']:.3f}.",
        "",
        "## Diagnosis",
        summary["diagnosis"],
        "",
        "## One Targeted Fix",
        summary["targeted_fix"],
        "",
        "## Do Not Run Next",
        summary["do_not_run_next"],
    ]
    (out_dir / "fold3_autopsy_report.md").write_text("\n".join(lines) + "\n")


def build_summary_from_json(out_dir):
    subset_payload = json.loads((out_dir / "subset_confusion_fold3.json").read_text())
    per_cause = json.loads((out_dir / "per_cause_metrics_fold3.json").read_text())
    faith = json.loads((out_dir / "mask_label_faithfulness_fold3.json").read_text())
    saturation = json.loads((out_dir / "noisy_or_saturation_fold3.json").read_text())
    primitive = json.loads((out_dir / "primitive_diff_disagreement_fold3.json").read_text())
    geometry = json.loads((out_dir / "primitive_geometry_by_fold.json").read_text())
    threshold = json.loads((out_dir / "threshold_probability_audit_fold3.json").read_text())
    map_recall = per_cause["evidence_map_noisy_or"]["per_cause"]["map"]["recall"]
    primitive_composed = per_cause["primitive_diff_rule"]["exact"]
    primitive_tuple = primitive["primitive_diff_rule_exact"]
    diagnosis = (
        "Fold3 is primarily an interaction-specific topology evidence/readout failure. "
        "Sensory is recovered perfectly and value/action do not overfire; "
        f"primitive-diff is {primitive_composed:.3f} on composed heldout, while tuple transitions expose "
        f"a {primitive_tuple:.3f} role-level primitive mismatch from sensory spillover in topology-only roles. "
        f"The learned failure comes from topology recall falling to {map_recall:.3f} when sensory and topology are composed."
    )
    targeted_fix = (
        "Test a cause-balanced / active-balanced mask loss that upweights topology-positive pixels in the sensory+topology fold."
    )
    do_not = (
        "Do not run broad architecture sweeps, larger transformers, new fold generation, or threshold-weakening experiments next."
    )
    return {
        "main_noisy": {
            "exact_fixed_0p5": threshold["exact_fixed_0p5"],
            "exact_val_exact_tuned": threshold["exact_val_exact_tuned"],
            "exact_val_f1_tuned": threshold["exact_val_f1_tuned"],
            "exact_heldout_exact_tuned_diagnostic": threshold["exact_heldout_exact_tuned_diagnostic"],
        },
        "subset_confusion": subset_payload,
        "per_cause": per_cause,
        "faithfulness": faith,
        "saturation": saturation,
        "primitive_disagreement": primitive,
        "geometry": geometry,
        "targeted_fix": targeted_fix,
        "do_not_run_next": do_not,
        "diagnosis": diagnosis,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="/private/tmp/final_evidence_map_modal_download/final_evidence_map_modal")
    ap.add_argument("--data-root", default="/private/tmp/iconip_autopsy_data")
    ap.add_argument("--out-dir", default="results/fold3_autopsy")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--finalize-only", action="store_true")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if args.device == "auto" and torch.backends.mps.is_available() else "cpu")
    print(f"fold3 autopsy using device={device}", flush=True)
    if args.finalize_only:
        fold3 = "fold3_sensory_topology"
        model = load_model(results_root, "evidence_map_noisy_or", fold3, 0, device)
        values = collect_model(model, data_root / fold3 / "test_composed_heldout.npz", device, include_grids=True)
        viz = make_visualizations(values, out_dir / "visualizations")
        write_json(out_dir / "visualization_index.json", viz)
        summary = build_summary_from_json(out_dir)
        summary["artifacts"] = {
            "results_root": str(results_root),
            "data_root": str(data_root),
            "out_dir": str(out_dir),
            "device": str(device),
            "visualization_seed": 0,
        }
        write_json(out_dir / "fold3_autopsy_summary.json", summary)
        summarize_report(out_dir, summary)
        print(json.dumps({"status": "complete", "mode": "finalize_only", "out_dir": str(out_dir)}, indent=2))
        return

    fold3 = "fold3_sensory_topology"
    split = data_root / fold3 / "test_composed_heldout.npz"
    val_split = data_root / fold3 / "val_composed_seen.npz"
    noisy_by_seed = {}
    aggregate = None
    for seed in (0, 1, 2):
        print(f"collect noisy fold3 seed={seed}", flush=True)
        model = load_model(results_root, "evidence_map_noisy_or", fold3, seed, device)
        vals = collect_model(model, split, device, include_grids=True)
        noisy_by_seed[seed] = vals
        del model
        gc.collect()
    aggregate = {
        key: np.concatenate([noisy_by_seed[s][key] for s in (0, 1, 2)])
        for key in noisy_by_seed[0]
        if isinstance(noisy_by_seed[0][key], np.ndarray)
    }
    noisy_threshold_by_seed = {
        seed: {"pred": noisy_by_seed[seed]["pred"].copy(), "target": noisy_by_seed[seed]["target"].copy()}
        for seed in (0, 1, 2)
    }

    subset_payload = {
        "fold": fold3,
        "expected_pair": "sensory+topology",
        "target_subset": "1010",
        "per_seed": {str(seed): subset_confusion(noisy_by_seed[seed]["pred"], noisy_by_seed[seed]["target"]) for seed in (0, 1, 2)},
        "aggregate": subset_confusion(aggregate["pred"], aggregate["target"]),
        "interpretation": "Fold3 errors are dominated by missing topology when sensory is present if 1010->1000 is the largest wrong bucket.",
    }
    write_json(out_dir / "subset_confusion_fold3.json", subset_payload)
    del noisy_by_seed
    gc.collect()

    per_cause = {"evidence_map_noisy_or": per_cause_metrics(aggregate["pred"], aggregate["target"])}
    model_comparison = {"evidence_map_noisy_or": subset_payload["aggregate"]}
    for model_key in ("learned_classifier_mask", "grid_token_transformer", "cell_graph_gnn"):
        vals = []
        for seed in (0, 1, 2):
            print(f"collect model comparison {model_key} seed={seed}", flush=True)
            model = load_model(results_root, model_key, fold3, seed, device)
            vals.append(collect_model(model, split, device))
            del model
            gc.collect()
        merged = {k: np.concatenate([v[k] for v in vals]) for k in vals[0] if isinstance(vals[0][k], np.ndarray)}
        per_cause[model_key] = per_cause_metrics(merged["pred"], merged["target"])
        model_comparison[model_key] = subset_confusion(merged["pred"], merged["target"])
        del vals, merged
        gc.collect()
    primitive_vals = collect_primitive(split)
    per_cause["primitive_diff_rule"] = per_cause_metrics(primitive_vals["pred"], primitive_vals["target"])
    model_comparison["primitive_diff_rule"] = subset_confusion(primitive_vals["pred"], primitive_vals["target"])
    write_json(out_dir / "per_cause_metrics_fold3.json", per_cause)
    write_json(out_dir / "model_comparison_fold3_errors.json", model_comparison)

    tuple_values_by_seed = {}
    for seed in (0, 1, 2):
        print(f"collect tuple noisy seed={seed}", flush=True)
        model = load_model(results_root, "evidence_map_noisy_or", fold3, seed, device)
        tuple_values_by_seed[seed] = collect_model(model, data_root / fold3 / "test_tuple_heldout.npz", device)
        del model
        gc.collect()
    tuple_agg = {
        key: np.concatenate([tuple_values_by_seed[s][key] for s in (0, 1, 2)])
        for key in tuple_values_by_seed[0]
        if isinstance(tuple_values_by_seed[0][key], np.ndarray)
    }
    write_json(out_dir / "role_level_metrics_fold3.json", {
        "per_seed": {str(seed): role_metrics(tuple_values_by_seed[seed]) for seed in (0, 1, 2)},
        "aggregate": role_metrics(tuple_agg),
    })
    del tuple_values_by_seed, tuple_agg
    gc.collect()

    faith = mask_label_faithfulness(aggregate)
    write_json(out_dir / "mask_label_faithfulness_fold3.json", faith)
    write_json(out_dir / "active_mask_quality_fold3.json", active_mask_quality(aggregate))
    by_fold = {}
    for fold in FOLDS:
        vals = []
        for seed in (0, 1, 2):
            print(f"collect mask quality {fold} seed={seed}", flush=True)
            model = load_model(results_root, "evidence_map_noisy_or", fold, seed, device)
            vals.append(collect_model(model, data_root / fold / "test_composed_heldout.npz", device))
            del model
            gc.collect()
        merged = {k: np.concatenate([v[k] for v in vals]) for k in vals[0] if isinstance(vals[0][k], np.ndarray)}
        by_fold[fold] = active_mask_quality(merged)
        del vals, merged
        gc.collect()
    write_json(out_dir / "mask_quality_by_fold.json", by_fold)

    threshold = threshold_probability_audit(aggregate, noisy_threshold_by_seed, results_root, fold3)
    write_json(out_dir / "threshold_probability_audit_fold3.json", threshold)
    saturation = saturation_audit(aggregate)
    write_json(out_dir / "noisy_or_saturation_fold3.json", saturation)
    primitive = primitive_disagreement(data_root / fold3 / "test_tuple_heldout.npz")
    write_json(out_dir / "primitive_diff_disagreement_fold3.json", primitive)
    geometry = geometry_by_fold(data_root)
    write_json(out_dir / "primitive_geometry_by_fold.json", geometry)
    viz = make_visualizations(aggregate, out_dir / "visualizations")
    write_json(out_dir / "visualization_index.json", viz)

    noisy_main = {
        "exact_fixed_0p5": per_cause["evidence_map_noisy_or"]["exact"],
        "exact_val_exact_tuned": threshold["exact_val_exact_tuned"],
        "exact_val_f1_tuned": threshold["exact_val_f1_tuned"],
        "exact_heldout_exact_tuned_diagnostic": threshold["exact_heldout_exact_tuned_diagnostic"],
    }
    map_recall = per_cause["evidence_map_noisy_or"]["per_cause"]["map"]["recall"]
    sensory_recall = per_cause["evidence_map_noisy_or"]["per_cause"]["sensory"]["recall"]
    diagnosis = (
        "Fold3 is primarily a cause-interaction evidence/readout failure: sensory recall stays high, "
        f"but topology recall is {map_recall:.3f} under the sensory+topology heldout pair. "
        "Primitive diff is near-perfect, so this is not a primitive-label mismatch. "
        "Faithfulness and saturation diagnostics separate whether the misses are weak-map or readout calibration artifacts."
    )
    targeted_fix = (
        "Test a cause-balanced / active-balanced mask loss that upweights topology-positive pixels in the sensory+topology fold."
    )
    do_not = (
        "Do not run broad architecture sweeps, larger transformers, or new fold generation before testing the targeted readout fix."
    )
    summary = {
        "main_noisy": noisy_main,
        "subset_confusion": subset_payload,
        "per_cause": per_cause,
        "faithfulness": faith,
        "saturation": saturation,
        "primitive_disagreement": primitive,
        "geometry": geometry,
        "targeted_fix": targeted_fix,
        "do_not_run_next": do_not,
        "diagnosis": diagnosis,
        "artifacts": {
            "results_root": str(results_root),
            "data_root": str(data_root),
            "out_dir": str(out_dir),
            "device": str(device),
        },
    }
    write_json(out_dir / "fold3_autopsy_summary.json", summary)
    summarize_report(out_dir, summary)
    print(json.dumps({"status": "complete", "out_dir": str(out_dir), **noisy_main}, indent=2))


if __name__ == "__main__":
    main()
