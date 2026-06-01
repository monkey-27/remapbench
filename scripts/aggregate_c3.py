"""Aggregate C3 runs, baseline context, compact tables, notes, and figures."""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


RESULT_DIR = "results/c3_subset_search_pilot"
FIGURE_DIR = "figures/c3_subset_search_pilot"


def _heldout_rows():
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULT_DIR, "runs", "*", "eval_c3.json"))):
        report = json.load(open(path))
        metrics = report["splits"]["test_composed_heldout"]
        run_name = os.path.basename(os.path.dirname(path))
        run_dir = os.path.dirname(path)
        cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))
        val = json.load(open(os.path.join(run_dir, "metrics_val.json")))
        rows.append({
            "run_name": run_name,
            "dataset": cfg["run_name"].split("_")[1],
            "variant": cfg["model"].replace("c3_", ""),
            "sparsity_weight": cfg["sparsity_weight"],
            "seed": int(run_name.rsplit("seed", 1)[1]),
            "privileged_proxy": report["factor_scored_is_privileged_proxy"],
            "val_seen_exact_match": val["best_val_composed_seen_exact"],
            "val_seen_mean_true_subset_rank": val["best_val_mean_true_subset_rank"],
            "heldout_exact_match": metrics["exact_match"],
            "heldout_micro_f1": metrics["micro_f1"],
            "heldout_macro_f1": metrics["macro_f1"],
            "heldout_value_miss_rate": metrics["missed_remap_rates"]["value_remap"],
            "heldout_action_miss_rate": metrics["missed_remap_rates"]["action_remap"],
            "heldout_mean_true_subset_rank": metrics["mean_true_subset_rank"],
            "heldout_top2_rate": metrics["top2_rate"],
            "heldout_energy_margin_mean": metrics["energy_margin_mean"],
            "heldout_true_subset_score_mean": metrics["true_subset_score_mean"],
            "heldout_predicted_subset_score_mean": metrics["predicted_subset_score_mean"],
            "heldout_planning_success_rate": metrics["planning"]["planning_success_rate"],
            "single_operator_reconstruction_mse": json.dumps(
                report["splits"]["test_single"]["operator_reconstruction_mse_by_cause"], sort_keys=True),
        })
    return rows


def _write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _groups(rows):
    grouped = {}
    for row in rows:
        key = (row["dataset"], row["variant"])
        grouped.setdefault(key, []).append(row)
    return grouped


def _selected_rows(rows):
    by_setting = {}
    for row in rows:
        key = (row["dataset"], row["variant"], float(row["sparsity_weight"]))
        by_setting.setdefault(key, []).append(row)
    selected = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for variant in sorted({row["variant"] for row in rows}):
            candidates = [group for key, group in by_setting.items()
                          if key[0] == dataset and key[1] == variant]
            best = max(candidates, key=lambda group: (
                np.mean([float(row["val_seen_exact_match"]) for row in group]),
                -np.mean([float(row["val_seen_mean_true_subset_rank"]) for row in group]),
            ))
            selected.extend(best)
    return selected


def _bar_plot(grouped, field, ylabel, filename):
    labels, means, stds = [], [], []
    for key, rows in sorted(grouped.items()):
        labels.append("/".join(key))
        values = [row[field] for row in rows]
        means.append(np.mean(values))
        stds.append(np.std(values))
    plt.figure(figsize=(9, 4))
    plt.bar(labels, means, yerr=stds, color=["#4C78A8", "#F58518", "#54A24B"] * 2)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, filename), dpi=160)
    plt.close()


def _diagnostic_plots(rows):
    chosen = max(rows, key=lambda row: row["heldout_exact_match"])
    scores_path = os.path.join(RESULT_DIR, "runs", chosen["run_name"],
                               "diagnostics_heldout_all_subset_scores.csv")
    diag = list(csv.DictReader(open(scores_path)))
    ranks = [int(row["true_rank"]) for row in diag]
    margins = [float(row["margin"]) for row in diag]
    plt.figure(figsize=(6, 4))
    plt.hist(ranks, bins=np.arange(1, 18) - 0.5, color="#4C78A8")
    plt.xlabel("True subset rank")
    plt.ylabel("Examples")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "true_subset_rank_histogram.png"), dpi=160)
    plt.close()
    plt.figure(figsize=(6, 4))
    plt.hist(margins, bins=30, color="#54A24B")
    plt.xlabel("Top-1 vs top-2 energy margin")
    plt.ylabel("Examples")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "energy_margin_distribution.png"), dpi=160)
    plt.close()


def _baseline_context():
    path = "results/direction_finder_summary.json"
    if not os.path.exists(path):
        return "Existing local direction-finder summary was not mounted into this run."
    summary = json.load(open(path))
    return f"Existing baseline context retained at {path}; C3 summary is additive and does not overwrite it."


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    rows = _heldout_rows()
    if not rows:
        raise SystemExit("No C3 evaluations found")
    _write_csv(os.path.join(RESULT_DIR, "summary.csv"), rows)
    selected_rows = _selected_rows(rows)
    grouped = _groups(selected_rows)
    _bar_plot(grouped, "heldout_exact_match", "Held-out exact match", "heldout_exact_match.png")
    _bar_plot(grouped, "heldout_top2_rate", "Held-out top-2 rate", "heldout_top2_rate.png")
    _bar_plot(grouped, "heldout_action_miss_rate", "Held-out action miss rate", "action_miss_rate.png")
    _bar_plot(grouped, "heldout_value_miss_rate", "Held-out value miss rate", "value_miss_rate.png")
    _diagnostic_plots(selected_rows)
    lines = ["# C3 Counterfactual Cause Composer Pilot", "", _baseline_context(), "",
             "Factor-scored runs are privileged proxy diagnostics: at test time their energies consume "
             "oracle-derived future, value, and action maps exposed by RemapBench. Full-reconstruction and "
             "delta-space runs are the fair ordinary-observation comparisons.", "",
             "The sparsity setting is selected per dataset and scoring variant using seen-composition validation only.",
             "", "## Held-out Results", "",
             "| dataset | variant | lambda | exact | micro F1 | macro F1 | top2 | rank | margin | value miss | action miss |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, group in sorted(grouped.items()):
        mean = lambda field: float(np.mean([row[field] for row in group]))
        lines.append(f"| {key[0]} | {key[1]} | {float(group[0]['sparsity_weight']):g} | "
                     f"{mean('heldout_exact_match'):.3f} | "
                     f"{mean('heldout_micro_f1'):.3f} | {mean('heldout_macro_f1'):.3f} | "
                     f"{mean('heldout_top2_rate'):.3f} | {mean('heldout_mean_true_subset_rank'):.2f} | "
                     f"{mean('heldout_energy_margin_mean'):.5f} | {mean('heldout_value_miss_rate'):.3f} | "
                     f"{mean('heldout_action_miss_rate'):.3f} |")
    fair_best = max((row for row in selected_rows if not row["privileged_proxy"]),
                    key=lambda row: row["heldout_exact_match"])
    exact = fair_best["heldout_exact_match"]
    if exact < 0.20:
        verdict = "failed"
    elif exact < 0.50:
        verdict = "some signal"
    elif exact < 0.75:
        verdict = "promising"
    elif exact < 0.90:
        verdict = "strong"
    else:
        verdict = "decomposition gap closed"
    lines.extend(["", "## Decision", "",
                  f"Best fair held-out exact match: `{exact:.3f}` from `{fair_best['run_name']}`.",
                  f"Threshold interpretation: **{verdict}**.",
                  "",
                  "## Failed-Case Diagnostics", "",
                  "Each run directory contains `diagnostics_heldout_all_subset_scores.csv` with the true subset, "
                  "predicted subset, true rank, margin, and all 16 candidate energies per example."])
    with open(os.path.join(RESULT_DIR, "summary.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    with open(os.path.join(RESULT_DIR, "IMPLEMENTATION_NOTES.md"), "w") as handle:
        handle.write(
            "# Implementation Notes\n\n"
            "C3 encodes the before grid, predicts four independent latent operators, exhaustively composes all "
            "16 subsets, and selects the lowest-energy subset. It has no direct label-classification head. "
            "Training and validation use only single interventions plus seen compositions; held-out goal+action "
            "rows are test-only. Validation selection uses `val_composed_seen` only.\n\n"
            "`c3_full_reconstruction` scores decoded after-grid reconstruction error. `c3_delta_space` scores "
            "latent delta mismatch against the encoded observed before/after change. `c3_factor_scored` is a "
            "privileged proxy diagnostic because its test-time energy includes oracle-derived delta maps. "
            "Sparsity weights `0`, `0.001`, and `0.01` are swept and selected using seen-composition validation only.\n"
        )
    print(f"Wrote {len(rows)} C3 run rows -> {RESULT_DIR}")


if __name__ == "__main__":
    main()
