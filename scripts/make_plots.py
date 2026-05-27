"""
Paper-style plots for ERPM pilot (including GatedERPM gates and planning).
Usage:
    python scripts/make_plots.py --config configs/smoke.yaml --run_dir results/smoke_gated_erpm
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
GATE_NAMES = ["sensory", "value", "map", "action"]


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_split_arrays(data_dir, split):
    path = os.path.join(data_dir, f"{split}.npz")
    if not os.path.exists(path):
        return None
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


# ---------------------------------------------------------------------------
# Error dissociation — full_visual_error vs future_error
# ---------------------------------------------------------------------------
def plot_error_dissociation_full_visual(data, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    iids = data["intervention_id"]
    fve  = data.get("full_visual_error", np.zeros(len(iids)))
    fe   = data["future_error"]
    for iid_val, iname in INTERVENTION_NAMES.items():
        mask = iids == iid_val
        if mask.sum() == 0: continue
        ax.scatter(fve[mask], fe[mask], s=8, alpha=0.5,
                   label=iname, color=ITYPE_COLORS.get(iname, "gray"))
    ax.set_xlabel("full_visual_error")
    ax.set_ylabel("future_error")
    ax.set_title("Error Dissociation: Full Visual vs Future")
    ax.legend(fontsize=7, markerscale=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120); plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Error dissociation — nuisance_error vs future_error
# ---------------------------------------------------------------------------
def plot_error_dissociation_nuisance(data, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    iids = data["intervention_id"]
    ne   = data.get("nuisance_error", data.get("sensory_error", np.zeros(len(iids))))
    fe   = data["future_error"]
    for iid_val, iname in INTERVENTION_NAMES.items():
        mask = iids == iid_val
        if mask.sum() == 0: continue
        ax.scatter(ne[mask], fe[mask], s=8, alpha=0.5,
                   label=iname, color=ITYPE_COLORS.get(iname, "gray"))
    ax.set_xlabel("nuisance_error")
    ax.set_ylabel("future_error")
    ax.set_title("Error Dissociation: Nuisance vs Future")
    ax.legend(fontsize=7, markerscale=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120); plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Gate activation by intervention type
# ---------------------------------------------------------------------------
def plot_gate_activation_by_intervention(eval_json, out_path):
    if eval_json is None:
        print(f"  [SKIP] gate_activation_by_intervention (no eval json)"); return
    gate_data = eval_json.get("overall", {}).get("mean_gates_per_intervention")
    if not gate_data:
        print(f"  [SKIP] gate_activation_by_intervention (no gate data in eval json)"); return

    itypes = list(gate_data.keys())
    mat = np.array([gate_data[k] for k in itypes])  # [n_types, 4]

    fig, ax = plt.subplots(figsize=(7, max(3, len(itypes) * 0.6 + 1)))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(GATE_NAMES, fontsize=9)
    ax.set_yticks(range(len(itypes))); ax.set_yticklabels(itypes, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("Mean Gate Activations per Intervention Type")
    for i in range(len(itypes)):
        for j in range(4):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120); plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Gate ablation failure rates
# ---------------------------------------------------------------------------
def plot_gate_ablation_failures(abl_json, out_path):
    """Plot failure rate + false_struct_remap + delta_future_mse per ablation mode."""
    if abl_json is None:
        print(f"  [SKIP] gate_ablation_failures (no ablation json)"); return
    modes = list(abl_json.keys())

    failure_rates  = [1.0 - abl_json[m]["overall"].get("exact_match", 0.0) for m in modes]
    false_struct   = [abl_json[m]["overall"].get("false_structural_remap_on_sensory", float("nan"))
                      for m in modes]
    df_mse         = [abl_json[m]["overall"].get("delta_future_mse", float("nan")) for m in modes]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    def _safe_nan(vals):
        return [v if not (isinstance(v, float) and v != v) else 0.0 for v in vals]

    def _bar(ax, vals, title, ylabel, ymax=None):
        colors = ["#2ca25f" if m == "oracle" else
                  "#dd4b39" if m == "all_zero" else
                  "#e6a817" if m == "all_one" else "#3b7ec2" for m in modes]
        clean = _safe_nan(vals)
        bars = ax.bar(modes, clean, color=colors)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        if ymax:
            ax.set_ylim(0, ymax)
        for bar, v in zip(bars, clean):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005 * (ymax or 1),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6)
        ax.tick_params(axis="x", labelrotation=25, labelsize=7)

    _bar(axes[0], failure_rates, "Failure rate (1 − exact_match)", "failure rate", ymax=1.0)
    _bar(axes[1], false_struct,  "False struct remap (sensory-only)", "rate", ymax=1.0)
    _bar(axes[2], df_mse,        "Δfuture MSE under ablation", "MSE")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120); plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Planning regret by intervention type
# ---------------------------------------------------------------------------
def plot_planning_regret(eval_json, abl_json, out_path):
    if eval_json is None:
        print(f"  [SKIP] planning_regret (no eval json)"); return

    items = {}
    pm = eval_json.get("overall", {}).get("planning")
    if pm:
        items["learned"] = pm.get("mean_step_regret", float("nan"))

    if abl_json:
        for mode in ["oracle", "all_zero", "all_one"]:
            if mode in abl_json:
                abl_pm = abl_json[mode].get("overall", {}).get("planning")
                if abl_pm:
                    items[f"ablation/{mode}"] = abl_pm.get("mean_step_regret", float("nan"))

    if not items:
        print(f"  [SKIP] planning_regret (no planning data)"); return

    names = list(items.keys())
    vals  = [items[n] for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, vals, color="#3b7ec2")
    ax.set_ylabel("Mean step regret vs oracle")
    ax.set_title("Planning Regret")
    plt.xticks(rotation=20, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120); plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# False structural remap on sensory samples
# ---------------------------------------------------------------------------
def plot_false_structural_remap(bl_eval, model_eval, out_path):
    names, rates = [], []
    for bname, res in bl_eval.items():
        val = res["overall"].get("false_structural_remap_on_sensory", float("nan"))
        names.append(bname)
        rates.append(val if not (isinstance(val, float) and val != val) else 0.0)
    if model_eval:
        val = model_eval["overall"].get("false_structural_remap_on_sensory", 0.0)
        names.append(model_eval.get("model", "gated_erpm"))
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
# Missed remap rates
# ---------------------------------------------------------------------------
def plot_missed_remap(bl_eval, model_eval, out_path):
    struct_labels = ["value_remap", "map_remap", "action_remap"]
    model_name = model_eval.get("model", "gated_erpm") if model_eval else None
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
# Multilabel confusion heatmap
# ---------------------------------------------------------------------------
def plot_multilabel_heatmap(bl_eval, model_eval, out_path):
    model_name = model_eval.get("model", "gated_erpm") if model_eval else None
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
    ax.set_xticks(range(len(TARGET_NAMES)))
    ax.set_xticklabels(TARGET_NAMES, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title("Per-label F1 Heatmap")
    for i in range(len(model_labels)):
        for j in range(len(TARGET_NAMES)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120); plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Sample gate predictions
# ---------------------------------------------------------------------------
def plot_sample_gate_predictions(data, run_dir, model_name, out_path, n=4, device="cpu"):
    try:
        import torch
        from models import build_model
    except ImportError:
        print("  [SKIP] torch not available"); return

    ckpt_path = os.path.join(run_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        print("  [SKIP] best.pt not found"); return

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(model_name)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    iids = data["intervention_id"]
    n_show = min(n, len(iids))

    fig, axes = plt.subplots(n_show, 5, figsize=(11, n_show * 2.2))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for row_i in range(n_show):
        si = row_i
        before = data["before_grid"][si].astype(np.float32)
        after  = data["after_grid"][si].astype(np.float32)
        x_in   = torch.from_numpy(np.concatenate([before, after], axis=0))[None]

        with torch.no_grad():
            out = model(x_in)

        mh_true = data["target_multihot"][si]
        pred_logits = out["remap_logits"][0].numpy()
        pred_gates  = (1 / (1 + np.exp(-pred_logits)))
        mh_pred = (pred_gates > 0.5).astype(int)

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

        if "delta_future" in out:
            df_pred = out["delta_future"][0, 0].numpy()
            df_true = data["delta_future"][si]
            vm = max(abs(df_true).max(), abs(df_pred).max(), 1e-6)
            ax[2].imshow(df_true, cmap="RdBu_r", vmin=-vm, vmax=vm, interpolation="nearest")
            ax[2].set_title("true Δfut", fontsize=7); ax[2].axis("off")
            ax[3].imshow(df_pred, cmap="RdBu_r", vmin=-vm, vmax=vm, interpolation="nearest")
            ax[3].set_title("pred Δfut", fontsize=7); ax[3].axis("off")
        else:
            ax[2].axis("off"); ax[3].axis("off")

        ax[4].axis("off")
        iname = INTERVENTION_NAMES.get(int(iids[si]), "?")
        lines = [iname]
        for gi, gn in enumerate(GATE_NAMES):
            t_v, p_v = int(mh_true[gi]), int(mh_pred[gi])
            mark = "✓" if t_v == p_v else "✗"
            lines.append(f"{mark} {gn}: true={t_v} pred={p_v} ({pred_gates[gi]:.2f})")
        ax[4].text(0.05, 0.95, "\n".join(lines), transform=ax[4].transAxes,
                   fontsize=6, va="top",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    fig.suptitle(f"Gate Predictions — {model_name}", fontsize=9)
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

    split      = args.split
    data_dir   = cfg["data_dir"]
    data       = load_split_arrays(data_dir, split)
    train_data = load_split_arrays(data_dir, "train_single") or data
    if data is None:
        data = train_data
    if data is None:
        print("No data found; skipping plots."); return

    model_eval  = _load_json(os.path.join(args.run_dir, f"eval_{split}.json"))
    bl_eval     = _load_json(os.path.join(args.run_dir, f"baseline_eval_{split}.json"))
    abl_eval    = _load_json(os.path.join(args.run_dir, f"gate_ablation_eval_{split}.json"))
    model_name  = model_eval.get("model", "gated_erpm") if model_eval else cfg.get("model", "gated_erpm")

    # Error dissociation plots
    plot_error_dissociation_full_visual(
        train_data,
        os.path.join(fig_dir, "error_dissociation_full_visual.png")
    )
    plot_error_dissociation_nuisance(
        train_data,
        os.path.join(fig_dir, "error_dissociation_nuisance.png")
    )

    # Gate activation heatmap
    plot_gate_activation_by_intervention(
        model_eval,
        os.path.join(fig_dir, "gate_activation_by_intervention.png")
    )

    # Gate ablation failures
    plot_gate_ablation_failures(
        abl_eval,
        os.path.join(fig_dir, "gate_ablation_failures.png")
    )

    # Planning regret
    plot_planning_regret(
        model_eval, abl_eval,
        os.path.join(fig_dir, "planning_regret.png")
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

    # Sample gate predictions
    plot_sample_gate_predictions(
        data, args.run_dir, model_name,
        os.path.join(fig_dir, "sample_gate_predictions.png"), n=4
    )

    print(f"\nAll plots saved in {fig_dir}/")


if __name__ == "__main__":
    main()
