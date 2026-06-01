"""Evaluate a trained C3 model with subset-search and score diagnostics."""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml

from models.c3 import CAUSE_NAMES, subset_ids
from remapbench.data import RemapDataset
from remapbench.planning import compute_planning_metrics
from scripts.c3_common import build_c3_from_config, factor_targets, make_loader, move_batch
from scripts.evaluate import _oracle_path_lens, compute_multilabel_metrics, get_device


SPLITS = ["test_single", "test_composed_seen", "test_composed_heldout", "test_larger", "test_noisy"]


def _subset_name(bits):
    return "".join(str(int(bit)) for bit in bits)


@torch.no_grad()
def evaluate_split(model, path, cfg, device, diagnostic_csv=None):
    raw_npz = np.load(path, allow_pickle=True)
    raw = {key: raw_npz[key] for key in raw_npz.files}
    loader = make_loader(RemapDataset(path), cfg)
    predictions, targets, ranks, margins, true_scores, pred_scores = [], [], [], [], [], []
    predicted_values, delta_future, delta_value, action_delta = [], [], [], []
    op_sums = {cause: 0.0 for cause in CAUSE_NAMES}
    op_counts = {cause: 0 for cause in CAUSE_NAMES}
    diagnostic_rows = []
    cursor = 0
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch["before_grid"], batch["after_grid"], factor_targets(batch))
        true_ids = subset_ids(batch["target_multihot"])
        order = out["scores"].argsort(dim=1)
        batch_ranks = (order == true_ids.unsqueeze(1)).nonzero()[:, 1] + 1
        sorted_scores = out["scores"].sort(dim=1).values
        selected_future = model._gather(out["candidate_future"], out["predicted_ids"])
        selected_value = model._gather(out["candidate_value"], out["predicted_ids"])
        selected_action = model._gather(out["candidate_action"], out["predicted_ids"])
        predictions.append(out["predicted_multihot"].cpu().numpy())
        targets.append(batch["target_multihot"].cpu().numpy())
        ranks.append(batch_ranks.cpu().numpy())
        margins.append((sorted_scores[:, 1] - sorted_scores[:, 0]).cpu().numpy())
        true_scores.append(out["scores"].gather(1, true_ids[:, None]).squeeze(1).cpu().numpy())
        pred_scores.append(sorted_scores[:, 0].cpu().numpy())
        delta_future.append(selected_future.cpu().numpy())
        delta_value.append(selected_value.cpu().numpy())
        action_delta.append(selected_action.cpu().numpy())
        predicted_values.append((batch["value_before"] + selected_value).cpu().numpy())
        observed_delta = out["z_after"] - out["z_before"]
        for cause_idx, cause in enumerate(CAUSE_NAMES):
            mask = batch["target_multihot"][:, cause_idx] > 0.5
            if mask.any():
                errors = (out["operator_updates"][mask, cause_idx] - observed_delta[mask]).square()
                op_sums[cause] += float(errors.mean(dim=(1, 2, 3)).sum())
                op_counts[cause] += int(mask.sum())
        if diagnostic_csv is not None:
            score_np = out["scores"].cpu().numpy()
            pred_np = out["predicted_multihot"].cpu().numpy()
            target_np = batch["target_multihot"].cpu().numpy()
            for row_idx in range(len(score_np)):
                row = {
                    "sample_index": cursor + row_idx,
                    "sample_id": int(batch["sample_id"][row_idx]),
                    "true_subset": _subset_name(target_np[row_idx]),
                    "predicted_subset": _subset_name(pred_np[row_idx]),
                    "true_rank": int(batch_ranks[row_idx]),
                    "margin": float(sorted_scores[row_idx, 1] - sorted_scores[row_idx, 0]),
                }
                row.update({f"score_{_subset_name(bits)}": float(score_np[row_idx, sid])
                            for sid, bits in enumerate(model.cause_subsets.cpu().numpy())})
                diagnostic_rows.append(row)
        cursor += len(true_ids)
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    ranks = np.concatenate(ranks)
    margins = np.concatenate(margins)
    true_scores = np.concatenate(true_scores)
    pred_scores = np.concatenate(pred_scores)
    overall = compute_multilabel_metrics(predictions, targets)
    overall.update({
        "mean_true_subset_rank": float(ranks.mean()),
        "median_true_subset_rank": float(np.median(ranks)),
        "top2_rate": float((ranks <= 2).mean()),
        "energy_margin_mean": float(margins.mean()),
        "energy_margin_median": float(np.median(margins)),
        "true_subset_score_mean": float(true_scores.mean()),
        "predicted_subset_score_mean": float(pred_scores.mean()),
        "operator_reconstruction_mse_by_cause": {
            cause: op_sums[cause] / op_counts[cause] if op_counts[cause] else None
            for cause in CAUSE_NAMES
        },
        "subset_confusion": {
            "labels": [_subset_name(bits) for bits in model.cause_subsets.cpu().numpy()],
            "matrix": np.histogram2d(
                (targets * np.array([8, 4, 2, 1])).sum(axis=1),
                (predictions * np.array([8, 4, 2, 1])).sum(axis=1),
                bins=np.arange(17) - 0.5,
            )[0].astype(int).tolist(),
        },
    })
    oracle_lens = raw.get("path_len_after")
    if oracle_lens is None:
        oracle_lens = _oracle_path_lens(raw, None)
    overall["planning"] = compute_planning_metrics(
        list(raw["after_grid"]), list(raw["start_xy"]), list(raw["goal_after_xy"]),
        [value[0] for value in np.concatenate(predicted_values)], oracle_lens, max_steps=50, penalty=50,
    )
    overall["delta_future_mse"] = float(np.mean((np.concatenate(delta_future) - raw["delta_future"][:, None]) ** 2))
    overall["delta_value_mse"] = float(np.mean((np.concatenate(delta_value) - raw["delta_value"][:, None]) ** 2))
    overall["action_delta_mse"] = float(np.mean((np.concatenate(action_delta) - raw["action_delta"]) ** 2))
    if diagnostic_csv is not None:
        os.makedirs(os.path.dirname(diagnostic_csv), exist_ok=True)
        with open(diagnostic_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=diagnostic_rows[0].keys())
            writer.writeheader()
            writer.writerows(diagnostic_rows)
    return overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = get_device(cfg.get("device", "auto"))
    model = build_c3_from_config(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model_state"])
    run_dir = os.path.dirname(args.checkpoint)
    report = {"model": cfg["model"], "checkpoint": args.checkpoint,
              "factor_scored_is_privileged_proxy": cfg["model"] == "c3_factor_scored", "splits": {}}
    for split in SPLITS:
        path = os.path.join(cfg["data_dir"], f"{split}.npz")
        diagnostic_csv = None
        if split == "test_composed_heldout":
            diagnostic_csv = os.path.join(run_dir, "diagnostics_heldout_all_subset_scores.csv")
        report["splits"][split] = evaluate_split(model, path, cfg, device, diagnostic_csv)
        metrics = report["splits"][split]
        print(f"{split}: exact={metrics['exact_match']:.3f} top2={metrics['top2_rate']:.3f} "
              f"rank={metrics['mean_true_subset_rank']:.2f} margin={metrics['energy_margin_mean']:.5f}")
    output = args.output or os.path.join(run_dir, "eval_c3.json")
    with open(output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved C3 evaluation -> {output}")


if __name__ == "__main__":
    main()
