"""Evaluate primitive evidence-mask cause prediction and diagnostics."""
import argparse
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import build_model
from remapbench.counterfactual_difference import (
    CAUSES, CounterfactualTransitionDataset, CounterfactualTupleDataset, ROLE_TO_ID, split_path,
)
from scripts.evaluate import compute_multilabel_metrics, get_device
from scripts.final_common import common_metadata, load_config, validate_checkpoint, write_json


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    return value


def _mean(values):
    return float(np.mean(values)) if values else None


def _subset_key(bits):
    return "".join(str(int(x)) for x in bits)


def _metrics_at_threshold(preds, targets, threshold):
    if np.isscalar(threshold):
        return compute_multilabel_metrics(preds, targets, threshold=float(threshold))
    preds_b = preds > np.asarray(threshold)[None, :]
    targets_b = targets > 0.5
    eps = 1e-8
    exact = float((preds_b == targets_b).all(axis=1).mean())
    per_label = {}
    tp_all = fp_all = fn_all = 0
    names = ["sensory_update", "value_remap", "map_remap", "action_remap"]
    for i, name in enumerate(names):
        tp = int((preds_b[:, i] & targets_b[:, i]).sum())
        fp = int((preds_b[:, i] & ~targets_b[:, i]).sum())
        fn = int((~preds_b[:, i] & targets_b[:, i]).sum())
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        per_label[name] = {
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "tp": tp, "fp": fp, "fn": fn,
        }
        tp_all += tp
        fp_all += fp
        fn_all += fn
    micro_p = tp_all / (tp_all + fp_all + eps)
    micro_r = tp_all / (tp_all + fn_all + eps)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + eps)
    return {
        "exact_match": exact,
        "micro_f1": float(micro_f1),
        "macro_f1": float(np.mean([row["f1"] for row in per_label.values()])),
        "per_label": per_label,
        "n_samples": int(len(targets)),
    }


def _tune_thresholds(preds, targets):
    grid = np.linspace(0.05, 0.95, 19)
    thresholds = []
    targets_b = targets > 0.5
    for k in range(targets.shape[1]):
        best_t, best_f1 = 0.5, -1.0
        for threshold in grid:
            pred_b = preds[:, k] > threshold
            tp = (pred_b & targets_b[:, k]).sum()
            fp = (pred_b & ~targets_b[:, k]).sum()
            fn = (~pred_b & targets_b[:, k]).sum()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            if f1 > best_f1:
                best_f1, best_t = float(f1), float(threshold)
        thresholds.append(best_t)
    return thresholds


def _mask_stats(pred, target, label_targets=None):
    pred_bin = pred > 0.5
    target_bin = target > 0.5
    inter = (pred_bin & target_bin).sum(axis=(0, 2, 3))
    union = (pred_bin | target_bin).sum(axis=(0, 2, 3))
    pred_sum = pred_bin.sum(axis=(0, 2, 3))
    target_sum = target_bin.sum(axis=(0, 2, 3))
    dice = (2 * inter + 1e-5) / (pred_sum + target_sum + 1e-5)
    iou = (inter + 1e-5) / (union + 1e-5)
    fp = np.maximum(pred_sum - inter, 0) / np.maximum(pred_sum, 1)
    per_sample_inter = (pred_bin & target_bin).sum(axis=(2, 3))
    per_sample_sum = pred_bin.sum(axis=(2, 3)) + target_bin.sum(axis=(2, 3))
    per_sample_dice = (2 * per_sample_inter + 1e-5) / (per_sample_sum + 1e-5)
    stats = {
        cause: {"dice": float(dice[i]), "iou": float(iou[i]), "false_positive_fraction": float(fp[i])}
        for i, cause in enumerate(CAUSES)
    }
    if label_targets is not None:
        labels = label_targets > 0.5
        for i, cause in enumerate(CAUSES):
            active = labels[:, i]
            inactive = ~active
            stats[cause]["active_only_dice"] = float(per_sample_dice[active, i].mean()) if active.any() else None
            stats[cause]["inactive_false_positive_evidence"] = (
                float(pred[:, i][inactive].mean()) if inactive.any() else None
            )
    return stats


@torch.no_grad()
def collect_split(model, path, cfg, device, tuple_roles=None):
    ds = CounterfactualTransitionDataset(path, roles=tuple_roles)
    loader = DataLoader(ds, batch_size=cfg.get("batch_size", 256))
    preds, targets, masks, mask_targets, roles = [], [], [], [], []
    inactive = []
    model.eval()
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(before_grid=batch["before_grid"], after_grid=batch["after_grid"])
        prob = torch.sigmoid(out["cause_logits"])
        preds.append(prob.cpu().numpy())
        targets.append(batch["target_multihot"].cpu().numpy())
        masks.append(out["evidence_maps"].cpu().numpy())
        mask_targets.append(batch["primitive_masks"].cpu().numpy())
        roles.extend(batch["transition_role_id"].cpu().numpy().tolist())
        inactive_mask = batch["target_multihot"] <= 0.5
        inactive.extend(out["evidence_maps"].mean(dim=(2, 3))[inactive_mask].cpu().numpy().tolist())
    return {
        "pred": np.concatenate(preds),
        "target": np.concatenate(targets),
        "masks": np.concatenate(masks),
        "mask_targets": np.concatenate(mask_targets),
        "roles": np.array(roles),
        "inactive": inactive,
    }


def evaluate_collected(values, val_thresholds=None):
    pred = values["pred"]
    target = values["target"]
    metrics = compute_multilabel_metrics(pred, target)
    heldout_thresholds = _tune_thresholds(pred, target)
    threshold_metrics = {
        "exact_fixed_0p5": metrics["exact_match"],
        "exact_val_tuned": (
            _metrics_at_threshold(pred, target, val_thresholds)["exact_match"]
            if val_thresholds is not None else None
        ),
        "exact_heldout_oracle_threshold_diagnostic": _metrics_at_threshold(
            pred, target, heldout_thresholds)["exact_match"],
        "val_tuned_thresholds": val_thresholds,
        "heldout_oracle_thresholds_diagnostic": heldout_thresholds,
    }
    role_metrics = {}
    roles_arr = values["roles"]
    for name, role_id in ROLE_TO_ID.items():
        idx = roles_arr == role_id
        if idx.any():
            role_metrics[name] = {
                **compute_multilabel_metrics(pred[idx], target[idx]),
                "exact_val_tuned": (
                    _metrics_at_threshold(pred[idx], target[idx], val_thresholds)["exact_match"]
                    if val_thresholds is not None else None
                ),
            }
    pred_b = pred > 0.5
    target_b = target > 0.5
    confusion = {}
    for t, p in zip(target_b, pred_b):
        key = f"{_subset_key(t)}->{_subset_key(p)}"
        confusion[key] = confusion.get(key, 0) + 1
    metrics.update({
        **threshold_metrics,
        "inactive_evidence_magnitude": _mean(values["inactive"]),
        "primitive_mask": _mask_stats(values["masks"], values["mask_targets"], target),
        "role_metrics": role_metrics,
        "subset_confusion": confusion,
    })
    return metrics


@torch.no_grad()
def context_diagnostics(model, path, cfg, device):
    loader = DataLoader(CounterfactualTupleDataset(path), batch_size=cfg.get("tuple_batch_size", 128))
    gaps, union_gaps = {cause: [] for cause in CAUSES}, {cause: [] for cause in CAUSES}
    for batch in loader:
        batch = _to_device(batch, device)
        outs = {name: model(before_grid=batch[name]["before_grid"], after_grid=batch[name]["after_grid"])
                for name in ("base_to_A", "base_to_B", "A_to_AB", "base_to_AB")}
        for k, cause in enumerate(CAUSES):
            active_b = batch["base_to_B"]["target_multihot"][:, k] > 0.5
            if active_b.any():
                gap = 1.0 - F.cosine_similarity(
                    outs["base_to_B"]["evidence_vectors"][active_b, k],
                    outs["A_to_AB"]["evidence_vectors"][active_b, k],
                    dim=1,
                )
                gaps[cause].extend(gap.cpu().numpy().tolist())
            for iso_name in ("base_to_A", "base_to_B"):
                active = batch[iso_name]["target_multihot"][:, k] > 0.5
                if active.any():
                    gap = 1.0 - F.cosine_similarity(
                        outs[iso_name]["evidence_vectors"][active, k],
                        outs["base_to_AB"]["evidence_vectors"][active, k],
                        dim=1,
                    )
                    union_gaps[cause].extend(gap.cpu().numpy().tolist())
    return {
        "context_gap_by_cause": {cause: _mean(values) for cause, values in gaps.items()},
        "mixed_union_gap_by_cause": {cause: _mean(values) for cause, values in union_gaps.items()},
        "symmetric_ba_available": False,
    }


def primitive_diff_rule(path):
    """Rule baseline from primitive channel diffs, not an oracle for target labels."""
    ds = CounterfactualTransitionDataset(path)
    preds, targets = [], []
    for item in ds:
        preds.append((item["primitive_masks"].flatten(1).sum(dim=1) > 0).float().numpy())
        targets.append(item["target_multihot"].numpy())
    return compute_multilabel_metrics(np.stack(preds), np.stack(targets))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    validate_checkpoint(cfg, checkpoint)
    model = build_model(
        cfg.get("model", "evidence_mask"),
        hidden_channels=cfg.get("hidden_channels", 48),
        use_diff_channels=cfg.get("use_diff_channels", True),
        label_from_evidence_pool=cfg.get("label_from_evidence_pool", False),
        label_readout=cfg.get("label_readout"),
        readout_tau=cfg.get("readout_tau", 0.5),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    val_thresholds = None
    try:
        val_values = collect_split(model, split_path(cfg["data_dir"], cfg.get("val_split", "val_composed_seen")), cfg, device)
        val_thresholds = _tune_thresholds(val_values["pred"], val_values["target"])
    except (OSError, ValueError, zipfile.BadZipFile):
        pass
    splits = {}
    for split in ("test_single", "test_composed_seen", "test_composed_heldout"):
        try:
            splits[split] = evaluate_collected(
                collect_split(model, split_path(cfg["data_dir"], split), cfg, device),
                val_thresholds=val_thresholds,
            )
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            splits[split] = {"status": "failed", "error": repr(exc)}
    tuple_path = split_path(cfg["data_dir"], cfg.get("tuple_test_split", "test_tuple_heldout"))
    try:
        splits["test_tuple_heldout_transitions"] = evaluate_collected(
            collect_split(model, tuple_path, cfg, device),
            val_thresholds=val_thresholds,
        )
        diagnostics = context_diagnostics(model, tuple_path, cfg, device)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        splits["test_tuple_heldout_transitions"] = {"status": "failed", "error": repr(exc)}
        diagnostics = {"status": "failed", "error": repr(exc), "symmetric_ba_available": False}
    primitive = {}
    for split in ("test_composed_heldout",):
        try:
            primitive[split] = primitive_diff_rule(split_path(cfg["data_dir"], split))
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            primitive[split] = {"status": "failed", "error": repr(exc)}
    try:
        primitive["test_tuple_heldout_transitions"] = primitive_diff_rule(tuple_path)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        primitive["test_tuple_heldout_transitions"] = {"status": "failed", "error": repr(exc)}
    report = {
        **common_metadata(cfg, args.config), "status": "complete",
        "checkpoint_used": args.checkpoint, "split_metrics": splits,
        "context_diagnostics": diagnostics,
        "primitive_diff_rule": primitive,
        "threshold_calibration": {
            "source_split": cfg.get("val_split", "val_composed_seen"),
            "val_tuned_thresholds": val_thresholds,
            "method": "per-label F1 grid search on validation predictions",
        },
        "readout_type": cfg.get("label_readout", "evidence_maxpool" if cfg.get("label_from_evidence_pool") else "learned_classifier"),
        "loss_weights": {
            key: cfg.get(key, 0.0)
            for key in (
                "primitive_mask_weight", "inactive_weight", "context_difference_weight",
                "mixed_union_weight", "map_union_weight", "map_context_weight",
            )
        },
    }
    output = args.output or os.path.join("results", cfg["run_name"], "eval_counterfactual_difference.json")
    write_json(output, report)
    write_json(os.path.join("results", cfg["run_name"], "context_diagnostics.json"), diagnostics)
    heldout = splits["test_composed_heldout"]
    rule = report["primitive_diff_rule"]["test_composed_heldout"]
    if heldout.get("status") == "failed":
        print(f"heldout failed: {heldout['error']}")
    else:
        print(f"heldout exact={heldout['exact_match']:.3f} macro={heldout['macro_f1']:.3f} "
              f"primitive_diff_rule={rule.get('exact_match')}")


if __name__ == "__main__":
    main()
