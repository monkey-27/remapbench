"""
Paper-style plots for ERPM pilot.
Usage:
    python scripts/make_plots.py --config configs/smoke.yaml --run_dir results/smoke_erpm
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

TARGET_NAMES = ["sensory_update", "value_remap", "map_remap", "action_remap"]
INTERVENTION_NAMES = {
    0: "sensory_nuisance", 1: "goal_relocation",
    2: "topology_change",  3: "action_change",  4: "composed",
}
ITYPE_COLORS = {
    "sensory_nuisance": "#e6a817",
    "goal_relocation":  "#3b7ec2",
    "topology_change":  "#2ca25f",
    "action_change":    "#dd4b39",
    "composed":         "#9467bd",
}


def load_eval(run_dir, split):
    path = os.path.join(run_dir, f"eval_{split}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_baseline_eval(run_dir, split):
    path = os.path.join(run_dir, f"baseline_eval_{split}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_split_arrays(data_dir, split):
    import numpy as np
    path = os.path.join(data_dir, f"{split}.npz")
    if not os.path.exists(path):
        return None
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


# ---------------------------------------------------------------------------
# Plot 1: Error dissociation scatter
# ---------------------------------------------------------------------------
def plot_error_dissociation(data, out_path):
    """Scatter of nuisance_error vs future_error, coloured by intervention."""
    fig, ax = plt.subplots(figsize=(6, 5))
    iids = data["intervention_id"]
    ne  = data.get("nuisance_error",   data.get("sensory_error",  np.zeros(len(iids))))
    fe  = data["future_error"]

    for iid_val, iname in INTERVENTION_NAMES.items():
        mask = iids == iid_val
        if mask.sum() == 0:
            continue
        ax.scatter(ne[mask], fe[mask], s=8, alpha=0.5, label=iname,
                   color=ITYPE_COLORS.get(iname, "gray"))

    ax.set_xlabel("nuisance_error")
    ax.set_ylabel("future_error")
    ax.set_title("Error Dissociation: Nuisance vs Future")
    ax.legend(fontsize=7, markerscale=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120); plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Baseline false structural remap rate on sensory samples
# ---------------------------------------------------------------------------
def plot_false_structural_remap(bl_eval, model_eval, out_path):
    names, rates = [], []
    for bname, res in bl_eval.items():
        val = res["overall"].get("false_structural_remap_on_sensory", float("nan"))
        names.append(bname)
        rates.append(val if not (isinstance(val, float) and val != val) else 0.0)
    if model_eval:
        val = model_eval["overall"].get("false_structural_remap_on_sensory", 0.0)
        names.append(model_eval.get("model", "erpm"))
        rates.append(val if not (isinstance(val, float) and val != val) else 0.0)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#dd4b39" if "oracle" not in n else "#2ca25f" for n in names]
    bars = ax.bar(names, rates, color=colors)
    ax.set_ylabel("False structural remap rate\n(sensory-only samples)")
    ax.set_title("False Structural Remap on Sensory Nuisance")
    ax.set_ylim(0, 1)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{rate:.2f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=25, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120); plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: Missed remap rates
# ---------------------------------------------------------------------------
def plot_missed_remap(bl_eval, model_eval, out_path):
    struct_labels = ["value_remap", "map_remap", "action_remap"]
    model_name = model_eval.get("model", "erpm") if model_eval else None
    all_results = {}
    for bname, res in bl_eval.items():
        mr = res["overall"].get("missed_remap_rates", {})
        all_results[bname] = [mr.get(l, float("nan")) for l in struct_labels]
    if model_eval:
        mr = model_eval["overall"].get("missed_remap_rates", {})
        all_results[model_name] = [mr.get(l, float("nan")) for l in struct_labels]

    x = np.arange(len(struct_labels))
    n_models = len(all_results)
    width = 0.7 / n_models
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (mname, vals) in enumerate(all_results.items()):
        offset = (i - n_models / 2 + 0.5) * width
        clean = [v if not (isinstance(v, float) and v != v) else 0.0 for v in vals]
        ax.bar(x + offset, clean, width, label=mname, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(struct_labels, fontsize=9)
    ax.set_ylabel("Missed remap rate")
    ax.set_title("Missed Remap Rates by Model / Baseline")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120); plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 4: Remap multilabel confusion heatmap
# ---------------------------------------------------------------------------
def plot_multilabel_heatmap(bl_eval, model_eval, out_path):
    """Show per-label F1 for all models as a heatmap."""
    model_name = model_eval.get("model", "erpm") if model_eval else None
    rows = {}
    for bname, res in bl_eval.items():
        pl = res["overall"].get("per_label", {})
        rows[bname] = [pl.get(l, {}).get("f1", 0.0) for l in TARGET_NAMES]
    if model_eval:
        pl = model_eval["overall"].get("per_label", {})
        rows[model_name] = [pl.get(l, {}).get("f1", 0.0) for l in TARGET_NAMES]

    mat = np.array(list(rows.values()))
    model_labels = list(rows.keys())

    fig, ax = plt.subplots(figsize=(7, max(3, len(model_labels) * 0.5 + 1)))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(len(TARGET_NAMES))); ax.set_xticklabels(TARGET_NAMES, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(model_labels))); ax.set_yticklabels(model_labels, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title("Per-label F1 Heatmap")
    for i in range(len(model_labels)):
        for j in range(len(TARGET_NAMES)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120); plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 5: Sample predictions
# ---------------------------------------------------------------------------
def plot_sample_predictions(data, run_dir, model_name, out_path, n=4, device="cpu"):
    """Show before/after grid, true/predicted delta_future/delta_value, labels."""
    try:
        import torch
        from remapbench.data import RemapDataset
        from models.erpm import build_model
    except ImportError:
        print("  [SKIP] torch not available for sample_predictions plot"); return

    # Load model
    ckpt_path = os.path.join(run_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        print("  [SKIP] best.pt not found"); return
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(model_name)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    iids = data["intervention_id"]
    n_show = min(n, len(iids))
    indices = list(range(n_show))

    fig, axes = plt.subplots(n_show, 6, figsize=(13, n_show * 2.0))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for row_i, si in enumerate(indices):
        before = data["before_grid"][si].astype(np.float32)
        after  = data["after_grid"][si].astype(np.float32)
        x_in   = torch.from_numpy(np.concatenate([before, after], axis=0))[None]

        with torch.no_grad():
            out = model(x_in)

        pred_df = out["delta_future"][0, 0].numpy()
        pred_dv = out["delta_value"][0, 0].numpy()
        true_df = data["delta_future"][si]
        true_dv = data["delta_value"][si]
        mh_true = data["target_multihot"][si]
        mh_pred = (torch.sigmoid(out["remap_logits"])[0].numpy() > 0.5).astype(int)

        def _grid_rgb(g):
            H, W = g.shape[1], g.shape[2]
            img = np.ones((H, W, 3))
            img[g[0] == 1] = [0.2, 0.2, 0.2]
            img[g[3] == 1] = [0.9, 0.85, 0.5]
            img[g[1] == 1] = [0.0, 0.9, 0.0]
            img[g[2] == 1] = [1.0, 0.2, 0.2]
            return img

        ax = axes[row_i]
        ax[0].imshow(_grid_rgb(before), interpolation="nearest")
        ax[0].set_title("before", fontsize=7); ax[0].axis("off")
        ax[1].imshow(_grid_rgb(after),  interpolation="nearest")
        ax[1].set_title("after",  fontsize=7); ax[1].axis("off")

        vm = max(abs(true_df).max(), abs(pred_df).max(), 1e-6)
        ax[2].imshow(true_df, cmap="RdBu_r", vmin=-vm, vmax=vm, interpolation="nearest")
        ax[2].set_title("true Δfut", fontsize=7); ax[2].axis("off")
        ax[3].imshow(pred_df, cmap="RdBu_r", vmin=-vm, vmax=vm, interpolation="nearest")
        ax[3].set_title("pred Δfut", fontsize=7); ax[3].axis("off")
        vm2 = max(abs(true_dv).max(), abs(pred_dv).max(), 1e-6)
        ax[4].imshow(true_dv, cmap="RdBu_r", vmin=-vm2, vmax=vm2, interpolation="nearest")
        ax[4].set_title("true Δval", fontsize=7); ax[4].axis("off")

        ax[5].axis("off")
        tl = " ".join(TARGET_NAMES[i] for i, v in enumerate(mh_true) if v)
        pl = " ".join(TARGET_NAMES[i] for i, v in enumerate(mh_pred) if v)
        iname = INTERVENTION_NAMES.get(int(iids[si]), "?")
        txt = f"{iname}\ntrue:  {tl or '(none)'}\npred: {pl or '(none)'}"
        ax[5].text(0.05, 0.8, txt, transform=ax[5].transAxes,
                   fontsize=6, va="top",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    fig.suptitle(f"Sample Predictions — {model_name}", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight"); plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--split",   default="test_single")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    fig_dir = os.path.join(args.run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    split = args.split
    data  = load_split_arrays(cfg["data_dir"], split)
    if data is None:
        # Fall back to train_single for scatter plot
        data = load_split_arrays(cfg["data_dir"], "train_single")
    if data is None:
        print("No data found; skipping plots."); return

    model_eval  = load_eval(args.run_dir, split)
    bl_eval     = load_baseline_eval(args.run_dir, split)
    model_name  = model_eval.get("model", "erpm") if model_eval else "erpm"

    plot_error_dissociation(
        load_split_arrays(cfg["data_dir"], "train_single") or data,
        os.path.join(fig_dir, "error_dissociation_scatter.png")
    )

    if bl_eval and model_eval:
        plot_false_structural_remap(
            bl_eval, model_eval,
            os.path.join(fig_dir, "baseline_false_structural_remap.png")
        )
        plot_missed_remap(
            bl_eval, model_eval,
            os.path.join(fig_dir, "missed_remap_rates.png")
        )
        plot_multilabel_heatmap(
            bl_eval, model_eval,
            os.path.join(fig_dir, "remap_confusion_or_multilabel_heatmap.png")
        )
    else:
        print("  [INFO] Eval JSON not found — skipping model-comparison plots")

    if data:
        plot_sample_predictions(
            data, args.run_dir, model_name,
            os.path.join(fig_dir, "sample_predictions.png"), n=4
        )

    print(f"\nAll plots saved in {fig_dir}/")


if __name__ == "__main__":
    main()
