"""Evaluate FEPO on normal RemapBench splits and linked tuple splits."""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from models import build_model
from models.cpo import PATHWAYS
from remapbench.data import RemapDataset, TupleRemapDataset
from scripts.evaluate import compute_map_mse, compute_multilabel_metrics, evaluate_model, get_device
from scripts.evaluate_cpo import _active_recall, _append, _concat, _mse


EVIDENCE_KEYS = (
    "evidence_vectors",
    "pathway_evidence_features",
    "pathway_evidence",
    "pathway_evidences",
    "evidence_by_pathway",
    "evidence",
)


def _pathway_values(out, keys):
    """Return pathway-local tensors from a FEPO output when they are exposed."""
    containers = (
        out,
        out.get("composition_diagnostics", {}),
        out.get("tuple_composition_diagnostics", {}),
    )
    for container in containers:
        for key in keys:
            values = container.get(key)
            if isinstance(values, dict):
                if any(name in values for name in PATHWAYS):
                    return {name: values[name] for name in PATHWAYS if name in values}
            elif torch.is_tensor(values) and values.ndim >= 2 and values.shape[1] == len(PATHWAYS):
                return {name: values[:, idx] for idx, name in enumerate(PATHWAYS)}
    return None


def _mean_or_none(values):
    return float(np.mean(values)) if values else None


def _mean_terms(terms):
    return _mean_or_none([value for values in terms.values() for value in values])


def _per_pathway(terms):
    return {name: _mean_or_none(values) for name, values in terms.items()}


def _sigmoid(logits):
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


def _add_reuse_terms(terms, direct, single, targets):
    direct_updates = _pathway_values(direct, ("pathway_updates",))
    single_updates = _pathway_values(single, ("pathway_updates",))
    if direct_updates is None or single_updates is None:
        return False
    for idx, name in enumerate(PATHWAYS):
        if name not in direct_updates or name not in single_updates:
            continue
        mask = targets[:, idx] > 0.5
        if mask.any():
            terms[name].append(float(torch.mean(
                (direct_updates[name][mask] - single_updates[name][mask]).square())))
    return True


def _add_invariance_terms(terms, a, b, ab, target_a, target_b):
    a_evidence = _pathway_values(a, EVIDENCE_KEYS)
    b_evidence = _pathway_values(b, EVIDENCE_KEYS)
    ab_evidence = _pathway_values(ab, EVIDENCE_KEYS)
    if any(values is None for values in (a_evidence, b_evidence, ab_evidence)):
        return False
    for idx, name in enumerate(PATHWAYS):
        if not all(name in values for values in (a_evidence, b_evidence, ab_evidence)):
            continue
        active_a = target_a[:, idx] > 0.5
        active_b = target_b[:, idx] > 0.5
        if active_a.any():
            terms[name].append(float(torch.mean(
                (ab_evidence[name][active_a] - a_evidence[name][active_a]).square())))
        if active_b.any():
            terms[name].append(float(torch.mean(
                (ab_evidence[name][active_b] - b_evidence[name][active_b]).square())))
    return True


def _add_leakage_terms(terms, out, targets):
    evidence = _pathway_values(out, EVIDENCE_KEYS)
    if evidence is None:
        return False
    for idx, name in enumerate(PATHWAYS):
        if name not in evidence:
            continue
        inactive = targets[:, idx] <= 0.5
        if inactive.any():
            terms[name].append(float(torch.mean(evidence[name][inactive].square())))
    return True


def evaluate_tuple(model, loader, device, threshold):
    composed_store, direct_store = {}, {}
    a_store, b_store = {}, {}
    truths = {"target_multihot": [], "delta_future": [], "delta_value": [], "action_delta": []}
    reuse_terms = {name: [] for name in PATHWAYS}
    invariance_terms = {name: [] for name in PATHWAYS}
    leakage_terms = {name: [] for name in PATHWAYS}
    reuse_available = False
    evidence_available = False
    model.eval()
    with torch.no_grad():
        for batch in loader:
            a = batch["A"]["x"].to(device)
            b = batch["B"]["x"].to(device)
            ab = batch["AB"]["x"].to(device)
            target_a = batch["A"]["target_multihot"].to(device)
            target_b = batch["B"]["target_multihot"].to(device)
            target_ab = batch["AB"]["target_multihot"].to(device)
            target_base = batch["base"]["target_multihot"].to(device)
            out_base = model(batch["base"]["x"].to(device))
            out_comp = model.forward_tuple(a, b)
            out_direct = model(ab)
            out_a, out_b = model(a), model(b)
            reuse_available |= _add_reuse_terms(reuse_terms, out_direct, out_a, target_a)
            reuse_available |= _add_reuse_terms(reuse_terms, out_direct, out_b, target_b)
            evidence_available |= _add_invariance_terms(
                invariance_terms, out_a, out_b, out_direct, target_a, target_b)
            for out, target in (
                (out_base, target_base), (out_a, target_a), (out_b, target_b),
                (out_direct, target_ab),
            ):
                _add_leakage_terms(leakage_terms, out, target)
            _append(composed_store, out_comp)
            _append(direct_store, out_direct)
            _append(a_store, out_a)
            _append(b_store, out_b)
            for key in truths:
                truths[key].append(batch["AB"][key].numpy())
    composed, direct = _concat(composed_store), _concat(direct_store)
    single_a, single_b = _concat(a_store), _concat(b_store)
    truth = {key: np.concatenate(value, axis=0) for key, value in truths.items()}
    probs = _sigmoid(direct["remap_logits"])
    composed_probs = _sigmoid(composed["remap_logits"])
    probs_a = _sigmoid(single_a["remap_logits"])
    probs_b = _sigmoid(single_b["remap_logits"])
    predicted_or = np.logical_or(probs_a > threshold, probs_b > threshold)

    overall = compute_multilabel_metrics(probs, truth["target_multihot"], threshold)
    overall.update({f"{key}_mse": compute_map_mse(direct[key], truth[key])
                    for key in ("delta_future", "delta_value", "action_delta")})
    composed_overall = compute_multilabel_metrics(
        composed_probs, truth["target_multihot"], threshold)
    composed_overall.update({f"{key}_mse": compute_map_mse(composed[key], truth[key])
                             for key in ("delta_future", "delta_value", "action_delta")})
    diagnostics = {
        "evidence_invariance_error": _mean_terms(invariance_terms) if evidence_available else None,
        "evidence_invariance_error_by_pathway": (
            _per_pathway(invariance_terms) if evidence_available else None),
        "inactive_evidence_leakage": _mean_terms(leakage_terms),
        "inactive_evidence_leakage_by_pathway": _per_pathway(leakage_terms),
        "gate_or_consistency_error": float(((probs > threshold) != predicted_or).mean()),
        "operator_reuse_error": _per_pathway(reuse_terms) if reuse_available else None,
        "operator_reuse_mse": _mean_terms(reuse_terms) if reuse_available else None,
        "active_pathway_recall": _active_recall(probs, truth["target_multihot"], threshold),
    }
    composition_mse = _mse(
        composed, direct, ("z_updated", "delta_future", "delta_value", "action_delta"))
    if composition_mse:
        diagnostics["operator_composition_error"] = composition_mse
    return {
        "kind": "tuple",
        "overall": overall,
        "tuple_composed_overall": composed_overall,
        "fepo_diagnostics": diagnostics,
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
    model_name = args.model or cfg.get("model", "fepo")
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
    output = args.output or os.path.join("results", cfg["run_name"], "eval_fepo.json")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved FEPO evaluation -> {output}")


if __name__ == "__main__":
    main()
