"""Evaluate classification, residual quality, and tuple-slot retrieval."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import build_model
from models.slot_decomposition import PATHWAYS, SlotDecomposition
from remapbench.data import RemapDataset, TupleRemapDataset
from scripts.evaluate import compute_map_mse, compute_multilabel_metrics, get_device
from scripts.final_common import common_metadata, load_config, stable_pathway_metrics, validate_checkpoint, write_json


SPLITS = ("test_single", "test_composed_seen", "test_composed_heldout", "test_larger", "test_noisy")


def _to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _mean(values):
    return float(np.mean(values)) if values else None


def _pairwise_overlap(values):
    flat = F.normalize(values.flatten(2), dim=2)
    similarity = torch.einsum("bkc,blc->bkl", flat, flat)
    mask = ~torch.eye(similarity.shape[1], dtype=torch.bool, device=values.device)
    return similarity[:, mask].mean(dim=1)


@torch.no_grad()
def evaluate_direct(model, path, cfg, device):
    loader = DataLoader(RemapDataset(path), batch_size=cfg["batch_size"])
    store = {key: [] for key in (
        "pred", "target", "delta_future", "delta_value", "action_delta",
        "future_after", "value_after", "target_future_after", "target_value_after",
        "residual_total", "residual_target",
    )}
    inactive_norms, inactive_routes, attention_overlaps, slot_overlaps = [], [], [], []
    model.eval()
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(batch["x"])
        target = batch["target_multihot"]
        inactive = target <= 0.5
        residual_norm = out["slot_residuals"].square().mean(dim=(2, 3, 4)).sqrt()
        inactive_norms.extend(residual_norm[inactive].cpu().tolist())
        inactive_routes.extend(out["routing_weights"][inactive].cpu().tolist())
        attention_overlaps.extend(_pairwise_overlap(out["slot_attention"]).cpu().tolist())
        slot_overlaps.extend(_pairwise_overlap(out["slot_vectors"]).cpu().tolist())
        for key, value in (
            ("pred", torch.sigmoid(out["remap_logits"])), ("target", target),
            ("delta_future", out["delta_future"]), ("delta_value", out["delta_value"]),
            ("action_delta", out["action_delta"]), ("future_after", out["future_after"]),
            ("value_after", out["value_after"]), ("target_future_after", batch["future_after"]),
            ("target_value_after", batch["value_after"]), ("residual_total", out["residual_total"]),
            ("residual_target", SlotDecomposition.residual_target(batch)),
        ):
            store[key].append(value.cpu().numpy())
    values = {key: np.concatenate(parts) for key, parts in store.items()}
    overall = compute_multilabel_metrics(values["pred"], values["target"])
    metrics = stable_pathway_metrics({
        **overall,
        "delta_future_mse": compute_map_mse(values["delta_future"], values["residual_target"][:, 0:1]),
        "delta_value_mse": compute_map_mse(values["delta_value"], values["residual_target"][:, 1:2]),
        "action_delta_mse": compute_map_mse(values["action_delta"], values["residual_target"][:, 2:6]),
    })
    metrics.update({
        "future_after_mse": compute_map_mse(values["future_after"], values["target_future_after"]),
        "value_after_mse": compute_map_mse(values["value_after"], values["target_value_after"]),
        "global_residual_mse": compute_map_mse(values["residual_total"], values["residual_target"]),
        "inactive_residual_norm": _mean(inactive_norms),
        "inactive_routing_weight": _mean(inactive_routes),
        "attention_overlap": _mean(attention_overlaps),
        "slot_vector_cosine_overlap": _mean(slot_overlaps),
    })
    return metrics


@torch.no_grad()
def evaluate_tuple(model, path, cfg, device):
    loader = DataLoader(TupleRemapDataset(path), batch_size=cfg.get("tuple_batch_size", 32))
    retrieval, margins, positive_sims, wrong_sims, unrelated_sims = [], [], [], [], []
    retrieval_by_cause = {name: [] for name in PATHWAYS}
    component_errors = []
    model.eval()
    for batch in loader:
        a, b, ab = (_to_device(batch[key], device) for key in ("A", "B", "AB"))
        out_a, out_b, out_ab = model(a["x"]), model(b["x"]), model(ab["x"])
        for donor_out, donor in ((out_a, a), (out_b, b)):
            for index, name in enumerate(PATHWAYS):
                active = donor["target_multihot"][:, index] > 0.5
                if not active.any():
                    continue
                candidate = F.normalize(out_ab["slot_vectors"][active, index], dim=1)
                donor_slots = F.normalize(donor_out["slot_vectors"][active], dim=2)
                sims = torch.einsum("bc,bkc->bk", candidate, donor_slots)
                positive = sims[:, index]
                wrong = sims.masked_fill(
                    F.one_hot(torch.full_like(sims[:, 0].long(), index), len(PATHWAYS)).bool(),
                    -1e9,
                ).max(dim=1).values
                success = positive > wrong
                retrieval.extend(success.cpu().tolist())
                retrieval_by_cause[name].extend(success.cpu().tolist())
                margins.extend((positive - wrong).cpu().tolist())
                positive_sims.extend(positive.cpu().tolist())
                wrong_sims.extend(wrong.cpu().tolist())
                rolled = F.normalize(donor_out["slot_vectors"][active, index].roll(1, dims=0), dim=1)
                unrelated_sims.extend((candidate * rolled).sum(dim=1).cpu().tolist())
                target = SlotDecomposition.residual_target(donor)[active]
                component_errors.extend(
                    (out_ab["slot_residuals"][active, index] - target).square()
                    .mean(dim=(1, 2, 3)).cpu().tolist())
    return {
        "retrieval_top1": _mean(retrieval),
        "retrieval_margin": _mean(margins),
        "retrieval_by_cause": {key: _mean(value) for key, value in retrieval_by_cause.items()},
        "correct_donor_similarity": _mean(positive_sims),
        "wrong_donor_similarity": _mean(wrong_sims),
        "same_cause_unrelated_donor_similarity": _mean(unrelated_sims),
        "component_residual_mse": _mean(component_errors),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    validate_checkpoint(cfg, checkpoint)
    model = build_model("slot_decomposition", shared_pooled=cfg.get("shared_pooled", False)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    split_metrics = {
        split: evaluate_direct(model, os.path.join(cfg["data_dir"], f"{split}.npz"), cfg, device)
        for split in SPLITS
    }
    slot_diagnostics = None
    if cfg.get("tuple_test_split"):
        slot_diagnostics = evaluate_tuple(
            model, os.path.join(cfg["data_dir"], f"{cfg['tuple_test_split']}.npz"), cfg, device)
    report = {
        **common_metadata(cfg, args.config), "status": "complete",
        "checkpoint_used": args.checkpoint, "split_metrics": split_metrics,
        "slot_diagnostics": slot_diagnostics,
    }
    run_dir = os.path.join("results", cfg["run_name"])
    write_json(os.path.join(run_dir, "eval_slot_decomposition.json"), report)
    write_json(os.path.join(run_dir, "slot_diagnostics.json"), slot_diagnostics or {})
    heldout = split_metrics["test_composed_heldout"]
    print(f"heldout exact={heldout['exact_match']:.3f} macro={heldout['macro_f1']:.3f} "
          f"retrieval={slot_diagnostics['retrieval_top1'] if slot_diagnostics else None}")


if __name__ == "__main__":
    main()
