"""Evaluate matched cause-component patching on failed direct AB predictions."""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from models.cpo import PATHWAYS
from remapbench.data import TupleRemapDataset
from scripts.evaluate import get_device
from scripts.final_common import (
    TARGET_NAMES, common_metadata, load_config, validate_checkpoint, write_json,
)


def _exact(out, target):
    return ((torch.sigmoid(out["remap_logits"]) > 0.5) == (target > 0.5)).all(dim=1)


def _clone_updates(out):
    return {name: value.clone() for name, value in out["pathway_updates"].items()}


def _patched(model, recipient, donor_a, donor_b, target_a, target_b, missing, mode):
    gates = recipient["gates"].clone()
    updates = _clone_updates(recipient)
    batch = gates.shape[0]
    for row in range(batch):
        missing_slots = torch.nonzero(missing[row], as_tuple=False).flatten().tolist()
        for slot in missing_slots:
            donor = donor_a if target_a[row, slot] > 0.5 else donor_b
            donor_row = row
            dest = slot
            if mode == "random_sample_patch":
                donor_row = (row + 1) % batch
            elif mode == "wrong_pathway_patch":
                candidates = [idx for idx in range(len(PATHWAYS)) if idx not in missing_slots]
                if not candidates:
                    continue
                dest = candidates[0]
            source_name, dest_name = PATHWAYS[slot], PATHWAYS[dest]
            gates[row, dest] = donor["gates"][donor_row, slot]
            updates[dest_name][row] = donor["pathway_updates"][source_name][donor_row]
    return model.decode_from_components(recipient["z_before"], gates, updates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test_tuple_heldout")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    model = build_model(cfg["model"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    validate_checkpoint(cfg, checkpoint)
    model.load_state_dict(checkpoint["model_state"])
    if not hasattr(model, "decode_from_components"):
        raise ValueError(f"{cfg['model']} does not support component patching")
    dataset = TupleRemapDataset(os.path.join(cfg["data_dir"], f"{args.split}.npz"))
    if len(dataset) < 2:
        raise ValueError("Causal patching requires at least two tuples for unrelated-donor controls")
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    totals = {"n_total": 0, "n_direct_correct": 0, "n_direct_failed": 0,
              "n_failed_with_missing_pathway": 0}
    correct = {name: 0 for name in
               ("no_patch", "correct_full_component_patch", "wrong_pathway_patch",
                "random_sample_patch")}
    rescues = {name: 0 for name in correct if name != "no_patch"}
    missed_before = {name: 0 for name in TARGET_NAMES}
    missed_after = {name: 0 for name in TARGET_NAMES}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            target = batch["AB"]["target_multihot"].to(device)
            target_a = batch["A"]["target_multihot"].to(device)
            target_b = batch["B"]["target_multihot"].to(device)
            recipient = model(batch["AB"]["x"].to(device))
            donor_a = model(batch["A"]["x"].to(device))
            donor_b = model(batch["B"]["x"].to(device))
            direct_ok = _exact(recipient, target)
            missing = (target > 0.5) & (recipient["gates"] <= args.threshold)
            failed = ~direct_ok
            totals["n_total"] += len(target)
            totals["n_direct_correct"] += int(direct_ok.sum())
            totals["n_direct_failed"] += int(failed.sum())
            totals["n_failed_with_missing_pathway"] += int((failed & missing.any(dim=1)).sum())
            correct["no_patch"] += int(direct_ok.sum())
            for idx, name in enumerate(TARGET_NAMES):
                missed_before[name] += int((failed & missing[:, idx]).sum())
            for mode in rescues:
                out = _patched(model, recipient, donor_a, donor_b, target_a, target_b, missing, mode)
                ok = _exact(out, target)
                correct[mode] += int(ok.sum())
                rescues[mode] += int((failed & ok).sum())
                if mode == "correct_full_component_patch":
                    after_missing = (target > 0.5) & (out["gates"] <= args.threshold)
                    for idx, name in enumerate(TARGET_NAMES):
                        missed_after[name] += int((failed & after_missing[:, idx]).sum())
    failed_n = totals["n_direct_failed"]
    report = {
        **common_metadata(cfg, args.config), "checkpoint_used": args.checkpoint,
        "split": args.split, "patch_threshold": args.threshold, **totals,
        "direct_exact": correct["no_patch"] / totals["n_total"],
        "correct_patch_exact": correct["correct_full_component_patch"] / totals["n_total"],
        "correct_patch_rescue_rate_among_failed": rescues["correct_full_component_patch"] / failed_n if failed_n else None,
        "wrong_pathway_patch_exact": correct["wrong_pathway_patch"] / totals["n_total"],
        "wrong_pathway_rescue_rate_among_failed": rescues["wrong_pathway_patch"] / failed_n if failed_n else None,
        "random_patch_exact": correct["random_sample_patch"] / totals["n_total"],
        "random_patch_rescue_rate_among_failed": rescues["random_sample_patch"] / failed_n if failed_n else None,
        "missed_pathways_before": missed_before,
        "missed_pathways_after_correct_patch": missed_after,
        "status": "complete",
    }
    output = args.output or os.path.join("results", cfg["run_name"], "causal_patching.json")
    write_json(output, report)
    print(f"Saved causal patching -> {output}")


if __name__ == "__main__":
    main()
