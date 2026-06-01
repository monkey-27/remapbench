"""Measure direct AB versus privileged oracle-decomposed tuple composition."""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from remapbench.data import TupleRemapDataset
from scripts.evaluate import compute_multilabel_metrics, get_device
from scripts.final_common import common_metadata, load_config, validate_checkpoint, write_json


def _miss(overall):
    return {name: overall["per_label"][name]["fn"] for name in overall["per_label"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test_tuple_heldout")
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    validate_checkpoint(cfg, checkpoint)
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    loader = DataLoader(TupleRemapDataset(os.path.join(cfg["data_dir"], f"{args.split}.npz")),
                        batch_size=cfg.get("tuple_batch_size", 32))
    direct, oracle, truth = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            a, b, ab = (batch[name]["x"].to(device) for name in ("A", "B", "AB"))
            direct.append(torch.sigmoid(model(ab)["remap_logits"]).cpu().numpy())
            oracle.append(torch.sigmoid(model.forward_tuple(a, b)["remap_logits"]).cpu().numpy())
            truth.append(batch["AB"]["target_multihot"].numpy())
    truth = np.concatenate(truth)
    direct_metrics = compute_multilabel_metrics(np.concatenate(direct), truth)
    oracle_metrics = compute_multilabel_metrics(np.concatenate(oracle), truth)
    report = {
        **common_metadata(cfg, args.config),
        "checkpoint_used": args.checkpoint,
        "split": args.split,
        "diagnostic_label": "privileged_oracle_decomposed_upper_bound_not_fair_baseline",
        "direct_AB_exact": direct_metrics["exact_match"],
        "direct_AB_macro_f1": direct_metrics["macro_f1"],
        "oracle_tuple_exact": oracle_metrics["exact_match"],
        "oracle_tuple_macro_f1": oracle_metrics["macro_f1"],
        "tuple_direct_gap_exact": oracle_metrics["exact_match"] - direct_metrics["exact_match"],
        "tuple_direct_gap_macro_f1": oracle_metrics["macro_f1"] - direct_metrics["macro_f1"],
        "direct_miss_by_pathway": _miss(direct_metrics),
        "oracle_miss_by_pathway": _miss(oracle_metrics),
        "status": "complete",
    }
    output = args.output or os.path.join("results", cfg["run_name"], "oracle_tuple_gap.json")
    write_json(output, report)
    print(f"Saved oracle tuple gap -> {output}")


if __name__ == "__main__":
    main()
