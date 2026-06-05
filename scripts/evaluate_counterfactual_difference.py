"""Evaluate counterfactual-difference cause prediction and diagnostics."""
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


def _mask_stats(pred, target):
    pred_bin = pred > 0.5
    target_bin = target > 0.5
    inter = (pred_bin & target_bin).sum(axis=(0, 2, 3))
    union = (pred_bin | target_bin).sum(axis=(0, 2, 3))
    pred_sum = pred_bin.sum(axis=(0, 2, 3))
    target_sum = target_bin.sum(axis=(0, 2, 3))
    dice = (2 * inter + 1e-5) / (pred_sum + target_sum + 1e-5)
    iou = (inter + 1e-5) / (union + 1e-5)
    fp = np.maximum(pred_sum - inter, 0) / np.maximum(pred_sum, 1)
    return {
        cause: {"dice": float(dice[i]), "iou": float(iou[i]), "false_positive_fraction": float(fp[i])}
        for i, cause in enumerate(CAUSES)
    }


@torch.no_grad()
def evaluate_split(model, path, cfg, device, tuple_roles=None):
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
    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    metrics = compute_multilabel_metrics(pred, target)
    role_metrics = {}
    roles_arr = np.array(roles)
    for name, role_id in ROLE_TO_ID.items():
        idx = roles_arr == role_id
        if idx.any():
            role_metrics[name] = compute_multilabel_metrics(pred[idx], target[idx])
    metrics.update({
        "inactive_evidence_magnitude": _mean(inactive),
        "primitive_mask": _mask_stats(np.concatenate(masks), np.concatenate(mask_targets)),
        "role_metrics": role_metrics,
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


def primitive_oracle(path):
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
        "counterfactual_difference",
        hidden_channels=cfg.get("hidden_channels", 48),
        use_diff_channels=cfg.get("use_diff_channels", True),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    splits = {}
    for split in ("test_single", "test_composed_seen", "test_composed_heldout"):
        try:
            splits[split] = evaluate_split(model, split_path(cfg["data_dir"], split), cfg, device)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            splits[split] = {"status": "failed", "error": repr(exc)}
    tuple_path = split_path(cfg["data_dir"], cfg.get("tuple_test_split", "test_tuple_heldout"))
    try:
        splits["test_tuple_heldout_transitions"] = evaluate_split(model, tuple_path, cfg, device)
        diagnostics = context_diagnostics(model, tuple_path, cfg, device)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        splits["test_tuple_heldout_transitions"] = {"status": "failed", "error": repr(exc)}
        diagnostics = {"status": "failed", "error": repr(exc), "symmetric_ba_available": False}
    primitive = {}
    for split in ("test_composed_heldout",):
        try:
            primitive[split] = primitive_oracle(split_path(cfg["data_dir"], split))
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            primitive[split] = {"status": "failed", "error": repr(exc)}
    try:
        primitive["test_tuple_heldout_transitions"] = primitive_oracle(tuple_path)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        primitive["test_tuple_heldout_transitions"] = {"status": "failed", "error": repr(exc)}
    report = {
        **common_metadata(cfg, args.config), "status": "complete",
        "checkpoint_used": args.checkpoint, "split_metrics": splits,
        "context_diagnostics": diagnostics,
        "primitive_oracle": primitive,
    }
    output = args.output or os.path.join("results", cfg["run_name"], "eval_counterfactual_difference.json")
    write_json(output, report)
    write_json(os.path.join("results", cfg["run_name"], "context_diagnostics.json"), diagnostics)
    heldout = splits["test_composed_heldout"]
    oracle = report["primitive_oracle"]["test_composed_heldout"]
    if heldout.get("status") == "failed":
        print(f"heldout failed: {heldout['error']}")
    else:
        print(f"heldout exact={heldout['exact_match']:.3f} macro={heldout['macro_f1']:.3f} "
              f"primitive_oracle={oracle.get('exact_match')}")


if __name__ == "__main__":
    main()
