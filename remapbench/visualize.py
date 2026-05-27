"""
Visualization script.
Usage:
    python -m remapbench.visualize --data_dir data/remapbench_v0 \
        --out_dir figures/remapbench_samples --n 8
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

INTERVENTION_NAMES = {0: "sensory_nuisance", 1: "goal_relocation",
                      2: "topology_change", 3: "action_change", 4: "composed"}
TARGET_NAMES = ["no_remap", "value_remap", "map_remap", "action_remap"]


def _grid_rgb(grid):
    """Render [C,H,W] grid to [H,W,3] RGB for display."""
    H, W = grid.shape[1], grid.shape[2]
    img = np.ones((H, W, 3))
    img[grid[0] == 1] = [0.2, 0.2, 0.2]      # wall: dark gray
    img[grid[3] == 1] = [0.9, 0.85, 0.5]      # nuisance: yellow
    img[grid[4] == 1] = [0.7, 0.5, 0.9]       # distractor: purple
    img[grid[5] == 1] = [0.4, 0.8, 0.4]       # door: green
    for a, color in enumerate(
        [[0.2, 0.7, 1.0], [0.2, 0.4, 1.0], [0.8, 0.4, 0.2], [1.0, 0.6, 0.2]]
    ):
        img[grid[6 + a] == 1] = color          # one-way: blue/orange tones
    img[grid[1] == 1] = [0.0, 0.9, 0.0]       # start: bright green
    img[grid[2] == 1] = [1.0, 0.2, 0.2]       # goal: red
    return img


def visualize_sample(sample_idx, data, ax_row):
    """Fill a row of axes for one sample."""
    iid = int(data["intervention_id"][sample_idx])
    iname = INTERVENTION_NAMES.get(iid, f"id{iid}")
    mh = data["target_multihot"][sample_idx]
    active_targets = [TARGET_NAMES[i] for i, v in enumerate(mh) if v]
    title_base = f"#{sample_idx} | {iname}\n{'+'.join(active_targets)}"

    before = data["before_grid"][sample_idx]
    after  = data["after_grid"][sample_idx]
    df     = data["delta_future"][sample_idx]
    dv     = data["delta_value"][sample_idx]
    adelta = data["action_delta"][sample_idx]  # [4, H, W]

    ax = ax_row

    ax[0].imshow(_grid_rgb(before), interpolation="nearest", vmin=0, vmax=1)
    ax[0].set_title("before", fontsize=7)
    ax[0].axis("off")

    ax[1].imshow(_grid_rgb(after), interpolation="nearest", vmin=0, vmax=1)
    ax[1].set_title("after", fontsize=7)
    ax[1].axis("off")

    vmax_f = max(abs(df).max(), 1e-6)
    ax[2].imshow(df, cmap="RdBu_r", vmin=-vmax_f, vmax=vmax_f, interpolation="nearest")
    ax[2].set_title("Δfuture", fontsize=7)
    ax[2].axis("off")

    vmax_v = max(abs(dv).max(), 1e-6)
    ax[3].imshow(dv, cmap="RdBu_r", vmin=-vmax_v, vmax=vmax_v, interpolation="nearest")
    ax[3].set_title("Δvalue", fontsize=7)
    ax[3].axis("off")

    # action_delta summary: max over actions
    adelta_sum = adelta.max(axis=0)
    ax[4].imshow(adelta_sum, cmap="hot", vmin=0, vmax=1, interpolation="nearest")
    ax[4].set_title("Δaction (max)", fontsize=7)
    ax[4].axis("off")

    # Annotation column
    ax[5].axis("off")
    se = float(data["sensory_error"][sample_idx])
    fe = float(data["future_error"][sample_idx])
    ve = float(data["value_error"][sample_idx])
    ae = float(data["action_error"][sample_idx])
    plb = int(data["path_len_before"][sample_idx])
    pla = int(data["path_len_after"][sample_idx])
    txt = (
        f"{title_base}\n"
        f"sensory={se:.3f}\n"
        f"future={fe:.3f}\n"
        f"value={ve:.3f}\n"
        f"action={ae:.3f}\n"
        f"path: {plb}→{pla}"
    )
    ax[5].text(0.05, 0.95, txt, transform=ax[5].transAxes,
               fontsize=6, va="top", ha="left",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))


def visualize(data_dir, out_dir, n=8):
    os.makedirs(out_dir, exist_ok=True)

    split_files = {
        "train_single": "train_single.npz",
        "test_composed": "test_composed.npz",
    }

    for split_name, fname in split_files.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found")
            continue

        data = {k: v for k, v in np.load(path, allow_pickle=True).items()}
        total = len(data["intervention_id"])
        n_show = min(n, total)
        indices = list(range(min(n_show, total)))

        n_cols = 6
        fig, axes = plt.subplots(n_show, n_cols, figsize=(n_cols * 2.0, n_show * 1.8))
        if n_show == 1:
            axes = axes[np.newaxis, :]

        for row_i, sample_idx in enumerate(indices):
            visualize_sample(sample_idx, data, axes[row_i])

        fig.suptitle(f"RemapBench – {split_name} (first {n_show} samples)", fontsize=9)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{split_name}_samples.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

        # Per-intervention-type figures for train_single
        if split_name == "train_single":
            iids = data["intervention_id"]
            for iid_val, iname in {0: "sensory_nuisance", 1: "goal_relocation",
                                   2: "topology_change", 3: "action_change"}.items():
                mask = np.where(iids == iid_val)[0]
                if len(mask) == 0:
                    continue
                n_show_type = min(n, len(mask))
                fig, axes = plt.subplots(n_show_type, 6, figsize=(12, n_show_type * 1.8))
                if n_show_type == 1:
                    axes = axes[np.newaxis, :]
                for row_i in range(n_show_type):
                    visualize_sample(int(mask[row_i]), data, axes[row_i])
                fig.suptitle(f"RemapBench – {iname}", fontsize=9)
                plt.tight_layout()
                op = os.path.join(out_dir, f"{iname}_samples.png")
                plt.savefig(op, dpi=100, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved {op}")


def main():
    parser = argparse.ArgumentParser(description="Visualize RemapBench samples")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()
    visualize(args.data_dir, args.out_dir, args.n)


if __name__ == "__main__":
    main()
