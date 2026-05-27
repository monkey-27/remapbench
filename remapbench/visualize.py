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

INTERVENTION_NAMES = {
    0: "sensory_nuisance", 1: "goal_relocation",
    2: "topology_change",  3: "action_change", 4: "composed",
}
TARGET_NAMES = ["sensory_update", "value_remap", "map_remap", "action_remap"]


def _grid_rgb(grid):
    """Render [C,H,W] uint8 grid to [H,W,3] float RGB."""
    H, W = grid.shape[1], grid.shape[2]
    img = np.ones((H, W, 3))
    img[grid[0] == 1] = [0.2, 0.2, 0.2]     # wall
    img[grid[3] == 1] = [0.9, 0.85, 0.5]    # nuisance_visual
    img[grid[4] == 1] = [0.7, 0.5, 0.9]     # distractor
    img[grid[5] == 1] = [0.4, 0.8, 0.4]     # door
    for a, color in enumerate(
        [[0.2, 0.7, 1.0], [0.2, 0.4, 1.0], [0.8, 0.4, 0.2], [1.0, 0.6, 0.2]]
    ):
        img[grid[6 + a] == 1] = color        # one-way channels
    img[grid[1] == 1] = [0.0, 0.9, 0.0]     # start
    img[grid[2] == 1] = [1.0, 0.2, 0.2]     # goal
    return img


def visualize_sample(idx, data, axrow):
    iid   = int(data["intervention_id"][idx])
    iname = INTERVENTION_NAMES.get(iid, f"id{iid}")
    mh    = data["target_multihot"][idx]
    active = [TARGET_NAMES[i] for i, v in enumerate(mh) if v]
    title  = f"#{idx} | {iname}\n{'+'.join(active)}"

    before = data["before_grid"][idx]
    after  = data["after_grid"][idx]
    df     = data["delta_future"][idx]
    dv     = data["delta_value"][idx]
    adelta = data["action_delta"][idx]

    ax = axrow
    ax[0].imshow(_grid_rgb(before), interpolation="nearest")
    ax[0].set_title("before", fontsize=7); ax[0].axis("off")
    ax[1].imshow(_grid_rgb(after),  interpolation="nearest")
    ax[1].set_title("after",  fontsize=7); ax[1].axis("off")

    vm_f = max(abs(df).max(), 1e-6)
    ax[2].imshow(df, cmap="RdBu_r", vmin=-vm_f, vmax=vm_f, interpolation="nearest")
    ax[2].set_title("Δfuture", fontsize=7); ax[2].axis("off")

    vm_v = max(abs(dv).max(), 1e-6)
    ax[3].imshow(dv, cmap="RdBu_r", vmin=-vm_v, vmax=vm_v, interpolation="nearest")
    ax[3].set_title("Δvalue", fontsize=7); ax[3].axis("off")

    ax[4].imshow(adelta.max(axis=0), cmap="hot", vmin=0, vmax=1, interpolation="nearest")
    ax[4].set_title("Δaction (max)", fontsize=7); ax[4].axis("off")

    ax[5].axis("off")
    ne = float(data["nuisance_error"][idx]) if "nuisance_error" in data else float(data.get("sensory_error", [0])[idx])
    fve = float(data["full_visual_error"][idx]) if "full_visual_error" in data else 0.0
    fe  = float(data["future_error"][idx])
    ve  = float(data["value_error"][idx])
    ae  = float(data["action_error"][idx])
    plb = int(data["path_len_before"][idx])
    pla = int(data["path_len_after"][idx])
    txt = (f"{title}\n"
           f"nuisance={ne:.3f}\nfull_vis={fve:.3f}\n"
           f"future={fe:.3f}\nvalue={ve:.3f}\naction={ae:.3f}\n"
           f"path: {plb}→{pla}")
    ax[5].text(0.05, 0.95, txt, transform=ax[5].transAxes, fontsize=6,
               va="top", ha="left",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))


def visualize(data_dir, out_dir, n=8):
    os.makedirs(out_dir, exist_ok=True)
    for split_name in ["train_single", "test_composed"]:
        path = os.path.join(data_dir, f"{split_name}.npz")
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found"); continue

        data    = {k: v for k, v in np.load(path, allow_pickle=True).items()}
        n_show  = min(n, len(data["intervention_id"]))
        indices = list(range(n_show))

        fig, axes = plt.subplots(n_show, 6, figsize=(12, n_show * 1.8))
        if n_show == 1:
            axes = axes[np.newaxis, :]
        for ri, si in enumerate(indices):
            visualize_sample(si, data, axes[ri])
        fig.suptitle(f"RemapBench — {split_name}", fontsize=9)
        plt.tight_layout()
        out = os.path.join(out_dir, f"{split_name}_samples.png")
        plt.savefig(out, dpi=100, bbox_inches="tight"); plt.close(fig)
        print(f"Saved {out}")

        if split_name == "train_single":
            iids = data["intervention_id"]
            for iid_val, iname in {0: "sensory_nuisance", 1: "goal_relocation",
                                   2: "topology_change", 3: "action_change"}.items():
                mask = np.where(iids == iid_val)[0]
                if len(mask) == 0: continue
                ns = min(n, len(mask))
                fig2, ax2 = plt.subplots(ns, 6, figsize=(12, ns * 1.8))
                if ns == 1: ax2 = ax2[np.newaxis, :]
                for ri in range(ns):
                    visualize_sample(int(mask[ri]), data, ax2[ri])
                fig2.suptitle(f"RemapBench — {iname}", fontsize=9)
                plt.tight_layout()
                op = os.path.join(out_dir, f"{iname}_samples.png")
                plt.savefig(op, dpi=100, bbox_inches="tight"); plt.close(fig2)
                print(f"Saved {op}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  required=True)
    parser.add_argument("--out_dir",   required=True)
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()
    visualize(args.data_dir, args.out_dir, args.n)


if __name__ == "__main__":
    main()
