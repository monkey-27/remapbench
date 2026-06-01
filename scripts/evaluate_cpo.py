"""Evaluate CPO on normal RemapBench splits and linked tuple splits."""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from models import build_model
from models.cpo import PATHWAYS
from remapbench.data import RemapDataset, TupleRemapDataset
from scripts.evaluate import compute_map_mse, compute_multilabel_metrics, evaluate_model, get_device


def _np(tensor):
    return tensor.detach().cpu().numpy()


def _concat(store):
    return {key: np.concatenate(value, axis=0) for key, value in store.items()}


def _append(store, out):
    for key in ("remap_logits", "gates", "delta_future", "delta_value", "action_delta",
                "value_after", "z_updated"):
        if key in out:
            store.setdefault(key, []).append(_np(out[key]))


def _mse(left, right, keys):
    return {f"{key}_mse": compute_map_mse(left[key], right[key])
            for key in keys if key in left and key in right}


def _active_recall(probs, targets, threshold):
    predicted, active = probs > threshold, targets > threshold
    names = ("sensory_update", "value_remap", "map_remap", "action_remap")
    return {name: float(predicted[active[:, idx], idx].mean()) if active[:, idx].any() else float("nan")
            for idx, name in enumerate(names)}


def evaluate_tuple(model, loader, device, threshold):
    composed_store, reverse_store, direct_store = {}, {}, {}
    a_store, b_store = {}, {}
    truths = {"target_multihot": [], "delta_future": [], "delta_value": [], "action_delta": []}
    reuse_terms = {name: [] for name in PATHWAYS}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            a = batch["A"]["x"].to(device)
            b = batch["B"]["x"].to(device)
            ab = batch["AB"]["x"].to(device)
            out_comp = model.forward_tuple(a, b)
            out_rev = model.forward_tuple(b, a)
            out_direct = model(ab)
            out_a, out_b = model(a), model(b)
            for idx, name in enumerate(PATHWAYS):
                mask_a = batch["A"]["target_multihot"][:, idx] > 0.5
                mask_b = batch["B"]["target_multihot"][:, idx] > 0.5
                if mask_a.any():
                    reuse_terms[name].append(float(F.mse_loss(
                        out_direct["pathway_updates"][name][mask_a.to(device)],
                        out_a["pathway_updates"][name][mask_a.to(device)])))
                if mask_b.any():
                    reuse_terms[name].append(float(F.mse_loss(
                        out_direct["pathway_updates"][name][mask_b.to(device)],
                        out_b["pathway_updates"][name][mask_b.to(device)])))
            _append(composed_store, out_comp)
            _append(reverse_store, out_rev)
            _append(direct_store, out_direct)
            _append(a_store, out_a)
            _append(b_store, out_b)
            for key in truths:
                truths[key].append(batch["AB"][key].numpy())
    composed, reverse, direct = _concat(composed_store), _concat(reverse_store), _concat(direct_store)
    single_a, single_b = _concat(a_store), _concat(b_store)
    truth = {key: np.concatenate(value, axis=0) for key, value in truths.items()}
    probs = 1.0 / (1.0 + np.exp(-composed["remap_logits"]))
    probs_rev = 1.0 / (1.0 + np.exp(-reverse["remap_logits"]))
    probs_direct = 1.0 / (1.0 + np.exp(-direct["remap_logits"]))
    probs_a = 1.0 / (1.0 + np.exp(-single_a["remap_logits"]))
    probs_b = 1.0 / (1.0 + np.exp(-single_b["remap_logits"]))
    predicted_or = np.logical_or(probs_a > threshold, probs_b > threshold)

    overall = compute_multilabel_metrics(probs, truth["target_multihot"], threshold)
    overall.update({f"{key}_mse": compute_map_mse(composed[key], truth[key])
                    for key in ("delta_future", "delta_value", "action_delta")})
    return {
        "kind": "tuple",
        "overall": overall,
        "cpo_diagnostics": {
            "gate_or_consistency_error": float(
                ((probs > threshold) != predicted_or).mean()),
            "operator_composition_error": _mse(
                composed, direct, ("z_updated", "delta_future", "delta_value", "action_delta")),
            "operator_reuse_error": {
                name: float(np.mean(values)) if values else float("nan")
                for name, values in reuse_terms.items()
            },
            "commutativity_error": _mse(
                composed, reverse, ("z_updated", "delta_future", "delta_value", "action_delta")),
            "active_pathway_recall": _active_recall(probs, truth["target_multihot"], threshold),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--splits", nargs="+")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output")
    args = parser.parse_args()
    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    device = get_device(cfg.get("device", "auto"))
    model_name = args.model or cfg.get("model", "cpo")
    model = build_model(model_name).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    data_dir = cfg["data_dir"]
    splits = args.splits or [
        os.path.splitext(os.path.basename(path))[0]
        for path in sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not os.path.basename(path).startswith(("train_", "val_"))
    ]
    report = {"model": model_name, "threshold": args.threshold, "splits": {}}
    for split in splits:
        path = os.path.join(data_dir, f"{split}.npz")
        if "_tuple_" in split:
            print(f"Evaluating tuple split {split}")
            loader = DataLoader(TupleRemapDataset(path), batch_size=cfg.get("tuple_batch_size", 32))
            report["splits"][split] = evaluate_tuple(model, loader, device, args.threshold)
        else:
            print(f"Evaluating normal split {split}")
            raw_npz = np.load(path, allow_pickle=True)
            raw = {key: raw_npz[key] for key in raw_npz.files}
            loader = DataLoader(RemapDataset(path), batch_size=cfg.get("batch_size", 128))
            overall, per_intervention = evaluate_model(
                model, loader, device, raw, threshold=args.threshold, compute_planning=True)
            report["splits"][split] = {
                "kind": "normal", "overall": overall, "per_intervention": per_intervention}
    output = args.output or os.path.join("results", cfg["run_name"], "eval_cpo.json")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved CPO evaluation -> {output}")


if __name__ == "__main__":
    main()
