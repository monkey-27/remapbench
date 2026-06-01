"""Evaluate strict causal patch controls without retraining a model."""
import argparse
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from models.cpo import PATHWAYS
from remapbench.data import TupleRemapDataset
from scripts.evaluate import compute_multilabel_metrics, get_device
from scripts.final_common import (
    TARGET_NAMES, common_metadata, load_config, validate_checkpoint, write_json,
)


CONDITIONS = (
    "no_patch",
    "oracle_gate_only",
    "correct_update_only_keep_original_gate",
    "correct_gate_plus_update",
    "same_pathway_unrelated_donor",
    "wrong_pathway_donor",
    "random_noise_update",
    "zero_missing_update",
)
OUTPUT_KEYS = (
    "delta_future", "delta_value", "action_delta", "future_after", "value_after",
)
RELEVANT_OUTPUT = {
    0: None,
    1: "delta_value",
    2: "delta_future",
    3: "action_delta",
}


def _clone_updates(out):
    return {name: value.clone() for name, value in out["pathway_updates"].items()}


def _validate_components(z_before, gates, updates):
    if gates.ndim != 2 or gates.shape[1] != len(PATHWAYS):
        raise ValueError(f"Expected gates [B,4], got {tuple(gates.shape)}")
    if gates.shape[0] != z_before.shape[0]:
        raise ValueError("Gate and latent batches differ")
    if gates.device != z_before.device:
        raise ValueError("Gate and latent devices differ")
    if set(updates) != set(PATHWAYS):
        raise ValueError(f"Unexpected pathway updates: {sorted(updates)}")
    for name, update in updates.items():
        if update.shape != z_before.shape:
            raise ValueError(f"{name} update shape {tuple(update.shape)} != {tuple(z_before.shape)}")
        if update.device != z_before.device:
            raise ValueError(f"{name} update device differs from latent device")


def _decode(model, recipient, gates, updates, original_logits=False):
    _validate_components(recipient["z_before"], gates, updates)
    out = model.decode_from_components(recipient["z_before"], gates, updates)
    if original_logits:
        out["remap_logits"] = recipient["remap_logits"]
    return out


def _matched_donor(donor_a, donor_b, target_a, row, slot):
    return donor_a if target_a[row, slot] > 0.5 else donor_b


def _pool_donors(model, loader, device):
    pools = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            tuple_ids = batch["tuple_id"].tolist()
            target_a = batch["A"]["target_multihot"]
            target_b = batch["B"]["target_multihot"]
            donors = (
                (model(batch["A"]["x"].to(device)), target_a),
                (model(batch["B"]["x"].to(device)), target_b),
            )
            for donor, targets in donors:
                for row, tuple_id in enumerate(tuple_ids):
                    for slot in torch.nonzero(targets[row] > 0.5, as_tuple=False).flatten().tolist():
                        pools[slot].append({
                            "tuple_id": int(tuple_id),
                            "gate": donor["gates"][row, slot].detach().cpu(),
                            "update": donor["pathway_updates"][PATHWAYS[slot]][row].detach().cpu(),
                        })
    return pools


def _unrelated(pools, tuple_id, slot, seed):
    donors = pools.get(slot, ())
    if not donors:
        return None
    start = (int(tuple_id) * 17 + slot * 31 + seed) % len(donors)
    for offset in range(len(donors)):
        donor = donors[(start + offset) % len(donors)]
        if donor["tuple_id"] != int(tuple_id):
            return donor
    return None


def _state_targets(batch, device):
    return {key: batch["AB"][key].to(device) for key in OUTPUT_KEYS}


def _collect(store, out, targets, target_multihot, failed, missing):
    store["preds"].append(torch.sigmoid(out["remap_logits"]).detach().cpu().numpy())
    store["targets"].append(target_multihot.detach().cpu().numpy())
    store["failed"].append(failed.detach().cpu().numpy())
    store["missing"].append(missing.detach().cpu().numpy())
    for key in OUTPUT_KEYS:
        store[key].append(out[key].detach().cpu().numpy())
        store[f"true_{key}"].append(targets[key].detach().cpu().numpy())


def _classification(preds, targets, failed):
    overall = compute_multilabel_metrics(preds, targets)
    pred_b, target_b = preds > 0.5, targets > 0.5
    result = {
        "exact_match": overall["exact_match"],
        "rescue_rate_among_failed": float(((pred_b == target_b).all(1) & failed).sum() / failed.sum())
        if failed.sum() else None,
    }
    for idx, name in enumerate(("sensory", "value", "map", "action")):
        positives, negatives = target_b[:, idx], ~target_b[:, idx]
        result[f"miss_{name}"] = float((~pred_b[positives, idx]).mean()) if positives.any() else None
        result[f"false_positive_{name}"] = (
            float(pred_b[negatives, idx].mean()) if negatives.any() else None)
    return result


def _mse(pred, target, mask=None):
    if mask is not None:
        if not mask.any():
            return None
        pred, target = pred[mask], target[mask]
    return float(np.mean((pred - target) ** 2))


def _metric(store):
    row = {key: np.concatenate(value) for key, value in store.items()}
    metrics = _classification(row["preds"], row["targets"], row["failed"])
    for key in OUTPUT_KEYS:
        metrics[f"{key}_mse"] = _mse(row[key], row[f"true_{key}"])
    return row, metrics


def _ratio_matches(control, correct):
    if correct is None or correct <= 0:
        return False
    return control is not None and control >= 0.8 * correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test_tuple_heldout")
    parser.add_argument("--output")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-tuples", type=int)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    if cfg["model"] not in ("gated_erpm", "factorized_gates"):
        raise ValueError(f"Unsupported strict patching model: {cfg['model']}")
    device = get_device(cfg.get("device", "auto"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    validate_checkpoint(cfg, checkpoint)
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = TupleRemapDataset(os.path.join(cfg["data_dir"], f"{args.split}.npz"))
    if args.max_tuples is not None:
        dataset = Subset(dataset, range(min(args.max_tuples, len(dataset))))
    if len(dataset) < 2:
        raise ValueError("Strict causal patching requires at least two tuples")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    donor_pools = _pool_donors(model, loader, device)
    stores = {condition: defaultdict(list) for condition in CONDITIONS}
    counts = {
        "n_total": 0, "n_direct_correct": 0, "n_direct_failed": 0,
        "n_failed_with_missing_pathway": 0,
        "n_missing_by_pathway": {name: 0 for name in TARGET_NAMES},
        "n_patched_by_condition": {name: 0 for name in CONDITIONS},
        "n_skipped_by_condition": {name: 0 for name in CONDITIONS},
    }

    with torch.no_grad():
        for batch in loader:
            tuple_ids = batch["tuple_id"].tolist()
            target = batch["AB"]["target_multihot"].to(device)
            target_a = batch["A"]["target_multihot"].to(device)
            recipient = model(batch["AB"]["x"].to(device))
            donor_a = model(batch["A"]["x"].to(device))
            donor_b = model(batch["B"]["x"].to(device))
            targets = _state_targets(batch, device)
            pred = torch.sigmoid(recipient["remap_logits"]) > 0.5
            failed = ~((pred == (target > 0.5)).all(dim=1))
            missing = (target > 0.5) & (recipient["gates"] <= args.threshold)
            counts["n_total"] += len(target)
            counts["n_direct_correct"] += int((~failed).sum())
            counts["n_direct_failed"] += int(failed.sum())
            counts["n_failed_with_missing_pathway"] += int((failed & missing.any(dim=1)).sum())
            for slot, name in enumerate(TARGET_NAMES):
                counts["n_missing_by_pathway"][name] += int(missing[:, slot].sum())
            _collect(stores["no_patch"], recipient, targets, target, failed, missing)

            for condition in CONDITIONS[1:]:
                gates = recipient["gates"].clone()
                updates = _clone_updates(recipient)
                original_logits = condition in (
                    "correct_update_only_keep_original_gate", "random_noise_update")
                for row in range(len(target)):
                    for slot in torch.nonzero(missing[row], as_tuple=False).flatten().tolist():
                        donor = _matched_donor(donor_a, donor_b, target_a, row, slot)
                        dest_name = PATHWAYS[slot]
                        if condition == "oracle_gate_only":
                            gates[row, slot] = donor["gates"][row, slot]
                        elif condition == "correct_update_only_keep_original_gate":
                            updates[dest_name][row] = donor["pathway_updates"][dest_name][row]
                        elif condition == "correct_gate_plus_update":
                            gates[row, slot] = donor["gates"][row, slot]
                            updates[dest_name][row] = donor["pathway_updates"][dest_name][row]
                        elif condition == "same_pathway_unrelated_donor":
                            unrelated = _unrelated(donor_pools, tuple_ids[row], slot, args.seed)
                            if unrelated is None:
                                counts["n_skipped_by_condition"][condition] += 1
                                continue
                            if unrelated["tuple_id"] == int(tuple_ids[row]):
                                raise ValueError("Unrelated donor reused recipient tuple_id")
                            gates[row, slot] = unrelated["gate"].to(device)
                            updates[dest_name][row] = unrelated["update"].to(device)
                        elif condition == "wrong_pathway_donor":
                            wrong_slot = (slot + 1) % len(PATHWAYS)
                            wrong_name = PATHWAYS[wrong_slot]
                            gates[row, slot] = donor["gates"][row, wrong_slot]
                            updates[dest_name][row] = donor["pathway_updates"][wrong_name][row]
                        elif condition == "random_noise_update":
                            source = donor["pathway_updates"][dest_name][row]
                            updates[dest_name][row] = (
                                torch.randn_like(source) * source.std(unbiased=False) * args.noise_scale
                                + source.mean())
                        elif condition == "zero_missing_update":
                            gates[row, slot] = donor["gates"][row, slot]
                            updates[dest_name][row] = torch.zeros_like(updates[dest_name][row])
                        counts["n_patched_by_condition"][condition] += 1
                out = _decode(model, recipient, gates, updates, original_logits=original_logits)
                _collect(stores[condition], out, targets, target, failed, missing)

    rows, condition_metrics = {}, {}
    for condition in CONDITIONS:
        rows[condition], condition_metrics[condition] = _metric(stores[condition])
    baseline = condition_metrics["no_patch"]
    for condition, metrics in condition_metrics.items():
        for key in ("delta_future", "delta_value", "action_delta"):
            metrics[f"{key}_mse_improvement"] = baseline[f"{key}_mse"] - metrics[f"{key}_mse"]
        metrics["pathway_specific_state_repair"] = {}
        for slot, name in enumerate(TARGET_NAMES):
            output = RELEVANT_OUTPUT[slot]
            if output is None:
                metrics["pathway_specific_state_repair"][name] = None
                continue
            mask = rows[condition]["missing"][:, slot]
            base_mse = _mse(rows["no_patch"][output], rows["no_patch"][f"true_{output}"], mask)
            patched_mse = _mse(rows[condition][output], rows[condition][f"true_{output}"], mask)
            metrics["pathway_specific_state_repair"][name] = (
                base_mse - patched_mse if base_mse is not None and patched_mse is not None else None)

    def _relevant_mean(condition):
        values = [value for value in
                  condition_metrics[condition]["pathway_specific_state_repair"].values()
                  if value is not None]
        return float(np.mean(values)) if values else None

    correct_state = _relevant_mean("correct_gate_plus_update")
    unrelated_state = _relevant_mean("same_pathway_unrelated_donor")
    wrong_state = _relevant_mean("wrong_pathway_donor")
    noise_state = _relevant_mean("random_noise_update")
    gate_only = condition_metrics["oracle_gate_only"]
    combined = condition_metrics["correct_gate_plus_update"]
    gate_state = _relevant_mean("oracle_gate_only")
    flags = {
        "gate_only_label_artifact_likely": (
            _ratio_matches(gate_only["rescue_rate_among_failed"], combined["rescue_rate_among_failed"])
            and (gate_state is None or gate_state <= 0)),
        "correct_patch_state_specific": (
            correct_state is not None and correct_state > 0
            and correct_state >= 2 * max(0.0, unrelated_state or 0.0)
            and correct_state >= 2 * max(0.0, wrong_state or 0.0)
            and correct_state >= 2 * max(0.0, noise_state or 0.0)),
        "same_pathway_unrelated_matches_correct": _ratio_matches(unrelated_state, correct_state),
        "wrong_pathway_low": wrong_state is None or wrong_state < 0.5 * max(0.0, correct_state or 0.0),
        "random_noise_low": noise_state is None or noise_state < 0.5 * max(0.0, correct_state or 0.0),
    }
    report = {
        **common_metadata(cfg, args.config),
        "config": args.config, "checkpoint": args.checkpoint, "split": args.split,
        "threshold": args.threshold, "seed": args.seed, "noise_scale": args.noise_scale,
        "wrong_pathway_convention": "patch next source pathway into missing destination pathway",
        "condition_metrics": condition_metrics, "summary_interpretation_flags": flags,
        **counts, "status": "complete",
    }
    output = args.output or os.path.join("results", cfg["run_name"], "causal_patching_v2.json")
    write_json(output, report)
    print(f"Saved strict causal patching -> {output}")


if __name__ == "__main__":
    main()
