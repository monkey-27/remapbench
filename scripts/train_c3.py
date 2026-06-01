"""Train one Counterfactual Cause Composer run on fair D1/D2 direct rows."""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml

from models.c3 import subset_ids
from scripts.c3_common import build_c3_from_config, factor_targets, load_splits, make_loader, move_batch
from scripts.evaluate import get_device


@torch.no_grad()
def evaluate_seen(model, loader, device):
    model.eval()
    correct = top2 = total = rank_sum = 0
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch["before_grid"], batch["after_grid"], factor_targets(batch))
        true_ids = subset_ids(batch["target_multihot"])
        order = out["scores"].argsort(dim=1)
        ranks = (order == true_ids.unsqueeze(1)).nonzero()[:, 1] + 1
        correct += int((out["predicted_ids"] == true_ids).sum())
        top2 += int((ranks <= 2).sum())
        rank_sum += int(ranks.sum())
        total += len(true_ids)
    return {"exact_match": correct / total, "top2_rate": top2 / total,
            "mean_true_subset_rank": rank_sum / total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--run-name-suffix", default="")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(cfg.get("device", "auto"))
    model = build_c3_from_config(cfg).to(device)
    train_ds = load_splits(cfg["data_dir"], cfg["train_splits"], args.max_train_samples)
    val_ds = load_splits(cfg["data_dir"], ["val_composed_seen"], args.max_val_samples)
    train_loader = make_loader(train_ds, cfg, shuffle=True)
    val_loader = make_loader(val_ds, cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
    run_name = f"{cfg['run_name']}{args.run_name_suffix}_seed{args.seed}"
    run_dir = os.path.join("results", run_name)
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))
    epochs = args.epochs or cfg.get("epochs", 30)
    print(f"Device: {device} | run={run_name} | train={len(train_ds)} | val_seen={len(val_ds)}")
    best_exact, best_rank = -1.0, float("inf")
    with open(os.path.join(run_dir, "train_log.jsonl"), "w") as handle:
        for epoch in range(1, epochs + 1):
            started = time.time()
            model.train()
            totals = {}
            n_samples = 0
            for batch in train_loader:
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                losses, _ = model.loss(batch)
                losses["total"].backward()
                optimizer.step()
                n_samples += len(batch["target_multihot"])
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach()) * len(batch["target_multihot"])
            metrics = evaluate_seen(model, val_loader, device)
            row = {"epoch": epoch, **{key: value / n_samples for key, value in totals.items()},
                   **{f"val_seen_{key}": value for key, value in metrics.items()}}
            ckpt = {"epoch": epoch, "model_state": model.state_dict(), "config": cfg,
                    "seed": args.seed, "val_seen": metrics}
            torch.save(ckpt, os.path.join(run_dir, "last.pt"))
            if metrics["exact_match"] > best_exact or (
                metrics["exact_match"] == best_exact and metrics["mean_true_subset_rank"] < best_rank
            ):
                best_exact, best_rank = metrics["exact_match"], metrics["mean_true_subset_rank"]
                torch.save(ckpt, os.path.join(run_dir, "best_val_composed_seen_exact.pt"))
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(f"Ep {epoch:3d} | loss={row['total']:.4f} | seen_exact={metrics['exact_match']:.3f} "
                  f"| top2={metrics['top2_rate']:.3f} | rank={metrics['mean_true_subset_rank']:.2f} "
                  f"| {time.time() - started:.1f}s")
    with open(os.path.join(run_dir, "metrics_val.json"), "w") as handle:
        json.dump({"best_val_composed_seen_exact": best_exact,
                   "best_val_mean_true_subset_rank": best_rank}, handle, indent=2)


if __name__ == "__main__":
    main()
