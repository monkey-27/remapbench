"""Modal topology-balance pilot for noisy-or primitive mask losses."""
import json
import os
import pathlib
import shutil
import subprocess
import time
from dataclasses import dataclass

import modal


ROOT = pathlib.Path(__file__).resolve().parents[1]
REMOTE = pathlib.Path("/root/iconip_monkey")
DATA_VOLUME_NAME = "iconip-monkey-final-decomp-gap-data"
OUTPUT_VOLUME_NAME = "iconip-monkey-final-evidence-map-results"
OUT_ROOT = pathlib.Path("results/topology_balance_pilot_modal")
PROFILE = os.environ.get("MODAL_PROFILE", "")
if PROFILE == "chat-arjunc":
    raise RuntimeError("Refusing to use disallowed Modal profile: chat-arjunc")

app = modal.App("iconip-monkey-topology-balance-pilot")
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml", "torch", "scikit-learn")
    .add_local_dir(ROOT / "models", REMOTE / "models")
    .add_local_dir(ROOT / "remapbench", REMOTE / "remapbench")
    .add_local_dir(ROOT / "scripts", REMOTE / "scripts")
    .add_local_dir(ROOT / "configs", REMOTE / "configs")
)


FOLDS = [
    {"fold_id": 0, "fold_name": "fold0_goal_topology", "heldout_pair_id": 0},
    {"fold_id": 1, "fold_name": "fold1_sensory_action", "heldout_pair_id": 1},
    {"fold_id": 2, "fold_name": "fold2_goal_action", "heldout_pair_id": 2},
    {"fold_id": 3, "fold_name": "fold3_sensory_topology", "heldout_pair_id": 3},
]
SEEDS = (0, 1, 2)
REQUIRED_SPLITS = (
    "train_single", "val_single", "test_single",
    "train_composed_seen", "val_composed_seen", "test_composed_seen",
    "test_composed_heldout", "train_tuple_seen", "val_tuple_seen",
    "test_tuple_seen", "test_tuple_heldout",
)


@dataclass(frozen=True)
class Variant:
    key: str
    display: str
    mask_loss_balancing: str
    active_cause_weight: float = 1.0
    inactive_cause_weight: float = 1.0
    positive_pixel_weight: float = 1.0
    mask_loss_normalize_by_active_causes: bool = False
    primitive_mask_weight: float = 1.0
    notes: str = ""


VARIANTS = [
    Variant(
        "noisy_or_mask_baseline",
        "noisy-or baseline mask loss",
        "none",
        notes="Current noisy-or primitive BCE+Dice mask loss.",
    ),
    Variant(
        "noisy_or_active_balanced_mask",
        "noisy-or active-balanced mask loss",
        "active_cause_balanced",
        inactive_cause_weight=0.25,
        mask_loss_normalize_by_active_causes=True,
        notes="Averages active causes equally and downweights inactive causes.",
    ),
    Variant(
        "noisy_or_pos_weighted_mask",
        "noisy-or positive-pixel weighted mask loss",
        "positive_pixel_weighted",
        positive_pixel_weight=5.0,
        notes="Weights positive mask pixels inside BCE.",
    ),
    Variant(
        "noisy_or_active_balanced_pos_weighted_mask",
        "noisy-or active-balanced + positive-pixel weighted mask loss",
        "active_cause_balanced_positive_pixel_weighted",
        inactive_cause_weight=0.25,
        positive_pixel_weight=5.0,
        mask_loss_normalize_by_active_causes=True,
        notes="Combines active-cause balancing with positive-pixel weighted BCE.",
    ),
]


def _remote_setup():
    os.chdir(REMOTE)
    out = pathlib.Path("/out/results")
    out.mkdir(parents=True, exist_ok=True)
    local_results = REMOTE / "results"
    if local_results.exists() and not local_results.is_symlink():
        shutil.rmtree(local_results)
    if not local_results.exists():
        local_results.symlink_to(out, target_is_directory=True)
    (local_results / "topology_balance_pilot_modal" / "runs").mkdir(parents=True, exist_ok=True)


def _run_python(code, env=None):
    subprocess.check_call(["python3", "-c", code], env=env)


@app.function(image=image, volumes={"/data": data_volume, "/out": output_volume}, timeout=20 * 60)
def inspect_data():
    _remote_setup()
    code = r'''
import json
from pathlib import Path
folds = ["fold0_goal_topology", "fold1_sensory_action", "fold2_goal_action", "fold3_sensory_topology"]
splits = ["train_single", "val_single", "test_single", "train_composed_seen", "val_composed_seen", "test_composed_seen", "test_composed_heldout", "train_tuple_seen", "val_tuple_seen", "test_tuple_seen", "test_tuple_heldout"]
candidates = [Path("/data/final_decomp_gap"), Path("/data"), Path("/data/data/final_decomp_gap")]
root = next((p for p in candidates if all((p / f).is_dir() for f in folds)), None)
if root is None:
    raise SystemExit("Could not find clean fold directories under: " + ", ".join(map(str, candidates)))
report = {"data_root": str(root), "folds": {}}
missing = []
for fold in folds:
    row = {}
    for split in splits:
        path = root / fold / f"{split}.npz"
        row[split] = {"exists": path.exists(), "path": str(path)}
        if not path.exists():
            missing.append(str(path))
    manifest = root / fold / "manifest.json"
    row["manifest"] = {"exists": manifest.exists(), "path": str(manifest)}
    report["folds"][fold] = row
if missing:
    raise SystemExit("Missing required fold files:\n" + "\n".join(missing))
out = Path("results/topology_balance_pilot_modal")
out.mkdir(parents=True, exist_ok=True)
(out / "data_inspection.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
'''
    _run_python(code)
    output_volume.commit()
    return "results/topology_balance_pilot_modal/data_inspection.json"


@app.function(image=image, volumes={"/data": data_volume, "/out": output_volume}, timeout=3 * 60 * 60)
def run_one(payload):
    _remote_setup()
    code = r'''
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
import yaml

sys.path.insert(0, "/root/iconip_monkey")
from models.counterfactual_difference import DiffCauseNet
from remapbench.counterfactual_difference import CounterfactualTransitionDataset, split_path
from scripts.evaluate_counterfactual_difference import (
    _metrics_at_threshold, _tune_thresholds, _tune_thresholds_exact,
    _tune_evidence_threshold, evaluate_collected, primitive_diff_rule,
)
from scripts.final_common import config_hash, commit_sha, write_json

payload = json.loads(os.environ["TOPOLOGY_BALANCE_PAYLOAD"])
fold = payload["fold"]
variant = payload["variant"]
seed = int(payload["seed"])
fold_name = fold["fold_name"]
model_key = variant["key"]
run_name = f"topology_balance_{model_key}_{fold_name}_seed{seed}"
root = Path("results/topology_balance_pilot_modal")
run_dir = root / "runs" / run_name
eval_name = f"eval_{model_key}_{fold_name}_seed{seed}.json"
if (run_dir / eval_name).exists():
    existing = json.loads((run_dir / eval_name).read_text())
    if existing.get("status") == "complete":
        print(json.dumps({"skip_complete": run_name}), flush=True)
        raise SystemExit(0)
run_dir.mkdir(parents=True, exist_ok=True)

folds = ["fold0_goal_topology", "fold1_sensory_action", "fold2_goal_action", "fold3_sensory_topology"]
candidates = [Path("/data/final_decomp_gap"), Path("/data"), Path("/data/data/final_decomp_gap")]
data_root = next((p for p in candidates if all((p / f).is_dir() for f in folds)), None)
if data_root is None:
    raise SystemExit("Could not find Modal data root")
data_dir = data_root / fold_name
cfg = {
    "model": "counterfactual_difference",
    "report_model_name": model_key,
    "model_key": model_key,
    "fold_id": fold["fold_id"],
    "fold_name": fold_name,
    "heldout_composed_pair_id": fold["heldout_pair_id"],
    "seed": seed,
    "data_dir": str(data_dir),
    "train_splits": ["train_single", "train_composed_seen"],
    "val_split": "val_composed_seen",
    "tuple_test_split": "test_tuple_heldout",
    "run_name": run_name,
    "batch_size": 256,
    "tuple_batch_size": 128,
    "epochs": 15,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "label_readout": "evidence_noisy_or",
    "readout_tau": 0.5,
    "readout_topk": 3,
    "primitive_mask_weight": float(variant["primitive_mask_weight"]),
    "mask_loss_balancing": variant["mask_loss_balancing"],
    "active_cause_weight": float(variant["active_cause_weight"]),
    "inactive_cause_weight": float(variant["inactive_cause_weight"]),
    "positive_pixel_weight": float(variant["positive_pixel_weight"]),
    "mask_loss_normalize_by_active_causes": bool(variant["mask_loss_normalize_by_active_causes"]),
    "inactive_weight": 0.0,
    "context_difference_weight": 0.0,
    "mixed_union_weight": 0.0,
    "map_context_weight": 0.0,
    "map_union_weight": 0.0,
    "use_diff_channels": True,
}
(run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
subprocess.check_call(["python3", "scripts/audit_evidence_mask.py", "--config", str(run_dir / "config.yaml"), "--out", str(run_dir / "audit_counterfactual_difference.json")])

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_model():
    return DiffCauseNet(
        label_readout="evidence_noisy_or",
        use_diff_channels=True,
        readout_tau=0.5,
        readout_topk=3,
    )

def primitive_mask_loss(evidence_maps, masks, target):
    balancing = cfg["mask_loss_balancing"]
    pos_weighted = "positive_pixel_weighted" in balancing
    bce = F.binary_cross_entropy(evidence_maps, masks, reduction="none")
    if pos_weighted:
        weights = torch.ones_like(masks)
        weights = torch.where(masks > 0.5, torch.full_like(weights, cfg["positive_pixel_weight"]), weights)
        bce = bce * weights
    bce = bce.mean(dim=(2, 3))
    inter = (evidence_maps * masks).sum(dim=(2, 3))
    denom = evidence_maps.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
    dice = 1.0 - (2.0 * inter + 1e-5) / (denom + 1e-5)
    per_cause = bce + dice
    if balancing in ("active_cause_balanced", "active_cause_balanced_positive_pixel_weighted"):
        active = target > 0.5
        inactive = ~active
        active_loss = (per_cause * active.float()).sum(dim=1) / active.float().sum(dim=1).clamp_min(1.0)
        inactive_loss = (per_cause * inactive.float()).sum(dim=1) / inactive.float().sum(dim=1).clamp_min(1.0)
        return (
            cfg["active_cause_weight"] * active_loss
            + cfg["inactive_cause_weight"] * inactive_loss
        ).mean()
    return per_cause.mean()

def train_model():
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device()
    model = make_model().to(dev)
    train_sets = [CounterfactualTransitionDataset(split_path(str(data_dir), split)) for split in cfg["train_splits"]]
    train_ds = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
    val_ds = CounterfactualTransitionDataset(split_path(str(data_dir), cfg["val_split"]))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best_exact, best_state, log_rows = -1.0, None, []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total, n = 0.0, 0
        started = time.time()
        for batch in train_loader:
            target = batch["target_multihot"].to(dev)
            before = batch["before_grid"].to(dev)
            after = batch["after_grid"].to(dev)
            masks = batch["primitive_masks"].to(dev)
            out = model(before_grid=before, after_grid=after)
            logits = out["cause_logits"]
            loss = F.binary_cross_entropy_with_logits(logits, target)
            loss = loss + cfg["primitive_mask_weight"] * primitive_mask_loss(out["evidence_maps"], masks, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * target.shape[0]
            n += target.shape[0]
        vals, tgts = [], []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                out = model(before_grid=batch["before_grid"].to(dev), after_grid=batch["after_grid"].to(dev))
                vals.append(torch.sigmoid(out["cause_logits"]).cpu().numpy())
                tgts.append(batch["target_multihot"].numpy())
        val = _metrics_at_threshold(np.concatenate(vals), np.concatenate(tgts), 0.5)
        if val["exact_match"] > best_exact:
            best_exact = val["exact_match"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        row = {
            "epoch": epoch,
            "train_loss": total / max(1, n),
            "val_exact": val["exact_match"],
            "val_macro_f1": val["macro_f1"],
            "seconds": time.time() - started,
        }
        log_rows.append(row)
        print(json.dumps({"run": run_name, **row}), flush=True)
    for row in log_rows:
        with (run_dir / "train_log.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
    ckpt = {
        "model_state": best_state,
        "config": cfg,
        "config_hash": config_hash(cfg),
        "model_key": model_key,
        "seed": seed,
        "fold_id": fold["fold_id"],
    }
    torch.save(ckpt, run_dir / "best_val_composed_seen_exact.pt")
    model.load_state_dict(best_state)
    return model

def collect_generic(model, split):
    dev = device()
    values = {"pred": [], "logits": [], "target": [], "masks": [], "mask_targets": [], "roles": [], "inactive": []}
    ds = CounterfactualTransitionDataset(split_path(str(data_dir), split))
    loader = DataLoader(ds, batch_size=cfg["batch_size"])
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model(before_grid=batch["before_grid"].to(dev), after_grid=batch["after_grid"].to(dev))
            logits = out["cause_logits"]
            maps = out["evidence_maps"].cpu().numpy()
            values["pred"].append(torch.sigmoid(logits).cpu().numpy())
            values["logits"].append(logits.cpu().numpy())
            values["target"].append(batch["target_multihot"].numpy())
            values["masks"].append(maps)
            values["mask_targets"].append(batch["primitive_masks"].numpy())
            values["roles"].extend(batch["transition_role_id"].numpy().tolist())
            inactive = batch["target_multihot"].numpy() <= 0.5
            values["inactive"].extend(maps.mean(axis=(2, 3))[inactive].tolist())
    return {
        "pred": np.concatenate(values["pred"]),
        "logits": np.concatenate(values["logits"]),
        "target": np.concatenate(values["target"]),
        "masks": np.concatenate(values["masks"]),
        "mask_targets": np.concatenate(values["mask_targets"]),
        "roles": np.array(values["roles"]),
        "inactive": values["inactive"],
    }

def topology_diagnostics(values):
    target = values["target"]
    pred = values["pred"]
    maps = values["masks"]
    mask_targets = values["mask_targets"]
    active_topology = target[:, 2] > 0.5
    topo = maps[active_topology, 2]
    topo_target = mask_targets[active_topology, 2]
    flat = topo.reshape(topo.shape[0], -1) if topo.size else np.zeros((0, 1), dtype=np.float32)
    top3 = np.sort(flat, axis=1)[:, -3:].mean(axis=1) if len(flat) else np.array([])
    pred_bin = topo > 0.5
    target_bin = topo_target > 0.5
    inter = (pred_bin & target_bin).sum(axis=(1, 2)) if len(topo) else np.array([])
    denom = pred_bin.sum(axis=(1, 2)) + target_bin.sum(axis=(1, 2)) if len(topo) else np.array([])
    dice = (2 * inter + 1e-5) / (denom + 1e-5) if len(topo) else np.array([])
    inactive_value = maps[target[:, 1] <= 0.5, 1]
    inactive_action = maps[target[:, 3] <= 0.5, 3]
    def map_stats(arr, prefix):
        if arr.size == 0:
            return {f"{prefix}_max_mean": None, f"{prefix}_top3_mean": None, f"{prefix}_area_gt_0p5_mean": None}
        arr_flat = arr.reshape(arr.shape[0], -1)
        return {
            f"{prefix}_max_mean": float(arr_flat.max(axis=1).mean()),
            f"{prefix}_top3_mean": float(np.sort(arr_flat, axis=1)[:, -3:].mean(axis=1).mean()),
            f"{prefix}_area_gt_0p5_mean": float((arr_flat > 0.5).mean(axis=1).mean()),
        }
    quantile_values = pred[active_topology, 2] if active_topology.any() else np.array([])
    diag = {
        **map_stats(topo, "topology_evidence"),
        **map_stats(inactive_value, "inactive_value_evidence"),
        **map_stats(inactive_action, "inactive_action_evidence"),
        "topology_active_dice_recomputed": float(dice.mean()) if len(dice) else None,
        "topology_positive_quantiles": (
            {str(q): float(np.percentile(quantile_values, q)) for q in (0, 5, 25, 50, 75, 95, 100)}
            if len(quantile_values) else {}
        ),
        "value_fp_count": int(((pred[:, 1] > 0.5) & (target[:, 1] <= 0.5)).sum()),
        "action_fp_count": int(((pred[:, 3] > 0.5) & (target[:, 3] <= 0.5)).sum()),
    }
    return diag

def eval_model(model):
    val_values = collect_generic(model, cfg["val_split"])
    val_f1 = _tune_thresholds(val_values["pred"], val_values["target"])
    val_exact = _tune_thresholds_exact(val_values["pred"], val_values["target"])
    val_evidence = _tune_evidence_threshold(val_values["masks"], val_values["target"], topk=3)
    splits = {}
    collected = {}
    for split in ("test_single", "test_composed_seen", "test_composed_heldout"):
        values = collect_generic(model, split)
        collected[split] = values
        splits[split] = evaluate_collected(
            values,
            val_f1_thresholds=val_f1,
            val_exact_thresholds=val_exact,
            val_evidence_threshold=val_evidence,
        )
    primitive = {"test_composed_heldout": primitive_diff_rule(split_path(str(data_dir), "test_composed_heldout"))}
    thresholds = {
        "source_split": cfg["val_split"],
        "val_f1_tuned_thresholds": val_f1,
        "val_exact_tuned_thresholds": val_exact,
        "val_evidence_threshold_topk3": val_evidence,
        "primary_metric": "exact_fixed_0p5",
        "secondary_metric": "exact_val_exact_tuned",
    }
    return splits, primitive, thresholds, topology_diagnostics(collected["test_composed_heldout"])

def audit_payload():
    audit_path = run_dir / "audit_counterfactual_difference.json"
    if not audit_path.exists():
        return {"passed": False, "errors": ["audit file missing"], "warnings": []}
    audit = json.loads(audit_path.read_text())
    return {
        "passed": audit.get("status") == "pass",
        "status": audit.get("status"),
        "errors": audit.get("errors", []),
        "warnings": audit.get("warnings", []),
        "summary": audit.get("summary", {}),
    }

started = time.time()
model = train_model()
splits, primitive, thresholds, fold3_diag = eval_model(model)
ckpt = str(run_dir / "best_val_composed_seen_exact.pt")
report = {
    "status": "complete",
    "run_name": run_name,
    "model_key": model_key,
    "display_name": variant["display"],
    "fold_id": fold["fold_id"],
    "fold_name": fold_name,
    "seed": seed,
    "split_metrics": splits,
    "primitive_diff_rule": primitive,
    "threshold_calibration": thresholds,
    "fold3_topology_diagnostics": fold3_diag,
    "audit": audit_payload(),
    "checkpoint_used": ckpt,
    "config_hash": config_hash(cfg),
    "commit": commit_sha(),
    "mask_loss_balancing": cfg["mask_loss_balancing"],
    "active_cause_weight": cfg["active_cause_weight"],
    "inactive_cause_weight": cfg["inactive_cause_weight"],
    "positive_pixel_weight": cfg["positive_pixel_weight"],
    "mask_loss_normalize_by_active_causes": cfg["mask_loss_normalize_by_active_causes"],
    "modal_resource": {"gpu": None, "function": "run_one"},
    "runtime_seconds": time.time() - started,
    "notes": variant.get("notes", ""),
}
write_json(run_dir / eval_name, report)
write_json(run_dir / "eval_counterfactual_difference.json", report)
if splits["test_composed_heldout"].get("faithfulness") is not None:
    write_json(run_dir / "faithfulness_audit.json", splits["test_composed_heldout"].get("faithfulness"))
manifest = {k: report[k] for k in ("run_name", "model_key", "display_name", "fold_id", "fold_name", "seed", "commit", "config_hash", "modal_resource", "runtime_seconds")}
manifest["command"] = "modal run scripts/modal_run_topology_balance_pilot.py"
write_json(run_dir / "manifest.json", manifest)
print(json.dumps({"done": run_name, "exact": splits["test_composed_heldout"].get("exact_fixed_0p5"), "topology_recall": splits["test_composed_heldout"].get("per_label", {}).get("map_remap", {}).get("recall")}), flush=True)
'''
    env = os.environ.copy()
    env["TOPOLOGY_BALANCE_PAYLOAD"] = json.dumps(payload)
    _run_python(code, env=env)
    output_volume.commit()
    return payload


@app.function(image=image, volumes={"/out": output_volume}, timeout=30 * 60)
def aggregate():
    _remote_setup()
    subprocess.check_call(["python3", "scripts/aggregate_topology_balance_pilot.py", "--root", str(OUT_ROOT)])
    output_volume.commit()
    return str(OUT_ROOT)


def _payloads():
    payloads = []
    for fold in FOLDS:
        for seed in SEEDS:
            for variant in VARIANTS:
                payloads.append({"fold": fold, "seed": seed, "variant": variant.__dict__})
    return payloads


def _launch():
    if not PROFILE:
        raise RuntimeError("Set MODAL_PROFILE explicitly before submission")
    print(f"Modal profile before submission: {PROFILE}")
    print(inspect_data.remote())
    jobs = _payloads()
    print(f"Submitting {len(jobs)} topology-balance pilot jobs")
    completed = list(run_one.map(jobs, return_exceptions=False))
    print(f"Completed {len(completed)} jobs")
    print(aggregate.remote())


@app.local_entrypoint()
def launch_full():
    _launch()


@app.local_entrypoint()
def aggregate_only():
    if not PROFILE:
        raise RuntimeError("Set MODAL_PROFILE explicitly before submission")
    print(f"Modal profile before aggregation: {PROFILE}")
    print(aggregate.remote())
