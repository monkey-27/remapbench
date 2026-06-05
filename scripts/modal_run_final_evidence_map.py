"""Modal final validation suite for evidence-map readout models."""
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
OUT_ROOT = pathlib.Path("results/final_evidence_map_modal")
PROFILE = os.environ.get("MODAL_PROFILE", "")
if PROFILE == "chat-arjunc":
    raise RuntimeError("Refusing to use disallowed Modal profile: chat-arjunc")

app = modal.App("iconip-monkey-final-evidence-map")
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
    family: str
    model_name: str | None = None
    readout: str | None = None
    primitive_mask_weight: float = 0.0
    use_diff_channels: bool = True
    train_splits: tuple[str, ...] = ("train_single", "train_composed_seen")
    notes: str = ""
    epochs: int = 15


TRAIN_VARIANTS = [
    Variant("standard_cnn", "Standard CNN", "direct_cnn", notes="Direct CNN using before/after/diff/absdiff input."),
    Variant("shared_gate", "Shared gate", "repo_model", model_name="gated_erpm", notes="Existing GatedERPM shared gate; before/after input."),
    Variant("independent_gate_head", "Independent/factorized gate head", "repo_model", model_name="factorized_gates", notes="Existing factorized gate heads; before/after input."),
    Variant("grid_token_transformer", "Grid-token transformer", "transformer", notes="Compact transformer over grid-cell tokens."),
    Variant("cell_graph_gnn", "Cell-graph GNN", "gnn", notes="Compact 4-neighbor message-passing baseline."),
    Variant("evidence_mask_learned_classifier", "Evidence-mask learned classifier", "evidence", readout="learned_classifier", primitive_mask_weight=1.0, notes="Mask supervision with learned hidden-feature classifier."),
    Variant("evidence_map_noisy_or", "Evidence-map noisy-or", "evidence", readout="evidence_noisy_or", primitive_mask_weight=1.0, notes="Final evidence-map method."),
    Variant("seen_pair_control", "Seen-pair control", "direct_cnn", train_splits=("train_single", "train_composed_seen", "test_composed_heldout"), notes="Privileged learnability control; heldout pair included in training."),
    Variant("bce_only", "BCE only", "evidence", readout="learned_classifier", primitive_mask_weight=0.0, notes="Evidence CNN without primitive mask supervision."),
    Variant("learned_classifier_mask", "learned classifier + mask", "evidence", readout="learned_classifier", primitive_mask_weight=1.0, notes="Ablation: masks without faithful evidence readout."),
    Variant("maxpool_mask", "maxpool + mask", "evidence", readout="evidence_maxpool", primitive_mask_weight=1.0, notes="Evidence maxpool readout ablation."),
    Variant("logsumexp_norm_mask", "logsumexp_norm + mask", "evidence", readout="evidence_logsumexp_norm", primitive_mask_weight=1.0, notes="Normalized logsumexp readout ablation."),
    Variant("topk3_mask", "topk3 + mask", "evidence", readout="evidence_topk", primitive_mask_weight=1.0, notes="Top-k evidence pooling with k=3."),
    Variant("noisy_or_mask", "noisy-or + mask", "evidence", readout="evidence_noisy_or", primitive_mask_weight=1.0, notes="Final method duplicate row for ablation table."),
    Variant("noisy_or_mask_no_diff", "noisy-or + mask, no diff channels", "evidence", readout="evidence_noisy_or", primitive_mask_weight=1.0, use_diff_channels=False, notes="Tests explicit diff-channel necessity."),
]
DIAGNOSTIC_VARIANTS = [
    Variant("primitive_diff_rule", "Primitive diff rule", "primitive_rule", notes="Structural primitive-diff diagnostic, not an oracle.", epochs=0),
    Variant("oracle_tuple_composer", "Oracle tuple composer", "oracle_tuple", notes="Privileged decomposed-tuple upper bound.", epochs=0),
    Variant("true_primitive_mask_readout", "true primitive mask readout", "true_mask", notes="Ceiling if primitive evidence masks are perfect.", epochs=0),
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
    (local_results / "final_evidence_map_modal" / "runs").mkdir(parents=True, exist_ok=True)


def _run_python(code):
    subprocess.check_call(["python3", "-c", code])


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
    report["folds"][fold] = row
if missing:
    raise SystemExit("Missing required split files:\n" + "\n".join(missing))
out = Path("results/final_evidence_map_modal")
out.mkdir(parents=True, exist_ok=True)
(out / "data_inspection.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
'''
    _run_python(code)
    output_volume.commit()
    return "results/final_evidence_map_modal/data_inspection.json"


@app.function(image=image, volumes={"/data": data_volume, "/out": output_volume}, timeout=3 * 60 * 60)
def run_one(payload):
    _remote_setup()
    code = r'''
import json, math, os, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
import yaml

sys.path.insert(0, "/root/iconip_monkey")
from models import build_model
from models.counterfactual_difference import DiffCauseNet
from remapbench.counterfactual_difference import CAUSES, CounterfactualTransitionDataset, split_path
from scripts.evaluate_counterfactual_difference import (
    _metrics_at_threshold, _tune_thresholds, _tune_thresholds_exact,
    _tune_evidence_threshold, evaluate_collected, collect_split, primitive_diff_rule,
)
from scripts.final_common import config_hash, commit_sha, write_json

payload = json.loads(os.environ["FINAL_PAYLOAD"])
fold = payload["fold"]
variant = payload["variant"]
seed = int(payload["seed"])
fold_name = fold["fold_name"]
model_key = variant["key"]
run_name = f"final_evidence_map_{model_key}_{fold_name}_seed{seed}"
root = Path("results/final_evidence_map_modal")
run_dir = root / "runs" / run_name
run_dir.mkdir(parents=True, exist_ok=True)

folds = ["fold0_goal_topology", "fold1_sensory_action", "fold2_goal_action", "fold3_sensory_topology"]
candidates = [Path("/data/final_decomp_gap"), Path("/data"), Path("/data/data/final_decomp_gap")]
data_root = next((p for p in candidates if all((p / f).is_dir() for f in folds)), None)
if data_root is None:
    raise SystemExit("Could not find Modal data root")
data_dir = data_root / fold_name
cfg = {
    "model": variant.get("model_name") or variant["family"],
    "report_model_name": model_key,
    "model_key": model_key,
    "fold_id": fold["fold_id"],
    "fold_name": fold_name,
    "heldout_composed_pair_id": fold["heldout_pair_id"],
    "seed": seed,
    "data_dir": str(data_dir),
    "train_splits": variant.get("train_splits", ["train_single", "train_composed_seen"]),
    "val_split": "val_composed_seen",
    "tuple_test_split": "test_tuple_heldout",
    "run_name": run_name,
    "batch_size": 256,
    "tuple_batch_size": 128,
    "epochs": int(variant.get("epochs", 15)),
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "label_readout": variant.get("readout"),
    "readout_tau": 0.5,
    "readout_topk": 3,
    "primitive_mask_weight": float(variant.get("primitive_mask_weight", 0.0)),
    "inactive_weight": 0.0,
    "context_difference_weight": 0.0,
    "mixed_union_weight": 0.0,
    "map_context_weight": 0.0,
    "map_union_weight": 0.0,
    "use_diff_channels": bool(variant.get("use_diff_channels", True)),
}
(run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
subprocess.check_call(["python3", "scripts/audit_evidence_mask.py", "--config", str(run_dir / "config.yaml"), "--out", str(run_dir / "audit_counterfactual_difference.json")])

class DirectCNN(nn.Module):
    def __init__(self, in_channels=40, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(hidden, 4),
        )
    def forward(self, before_grid=None, after_grid=None, x=None):
        if x is None:
            diff = after_grid - before_grid
            x = torch.cat([before_grid, after_grid, diff, diff.abs()], dim=1)
        return {"remap_logits": self.net(x)}

class GridTokenTransformer(nn.Module):
    def __init__(self, in_channels=40, hidden=96, heads=4, layers=3, max_tokens=256):
        super().__init__()
        self.proj = nn.Linear(in_channels, hidden)
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, hidden))
        enc = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(hidden, 4)
    def forward(self, before_grid=None, after_grid=None, x=None):
        diff = after_grid - before_grid
        x = torch.cat([before_grid, after_grid, diff, diff.abs()], dim=1)
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.proj(tokens) + self.pos[:, :tokens.shape[1]]
        return {"remap_logits": self.head(self.enc(tokens).mean(dim=1))}

class CellGraphGNN(nn.Module):
    def __init__(self, in_channels=40, hidden=96, layers=4):
        super().__init__()
        self.inp = nn.Conv2d(in_channels, hidden, 1)
        self.layers = nn.ModuleList(nn.Conv2d(hidden * 2, hidden, 1) for _ in range(layers))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(hidden, 4))
    def forward(self, before_grid=None, after_grid=None, x=None):
        diff = after_grid - before_grid
        h = self.inp(torch.cat([before_grid, after_grid, diff, diff.abs()], dim=1))
        for layer in self.layers:
            nbr = (
                torch.roll(h, 1, 2) + torch.roll(h, -1, 2)
                + torch.roll(h, 1, 3) + torch.roll(h, -1, 3)
            ) / 4.0
            h = F.relu(layer(torch.cat([h, nbr], dim=1)))
        return {"remap_logits": self.head(h)}

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_model():
    fam = variant["family"]
    if fam == "direct_cnn":
        return DirectCNN()
    if fam == "transformer":
        return GridTokenTransformer()
    if fam == "gnn":
        return CellGraphGNN()
    if fam == "repo_model":
        return build_model(variant["model_name"])
    if fam == "evidence":
        return DiffCauseNet(
            label_readout=variant["readout"],
            use_diff_channels=bool(variant.get("use_diff_channels", True)),
            readout_tau=0.5,
            readout_topk=3,
        )
    raise ValueError(f"trainable model not supported: {fam}")

def forward_logits(model, batch, dev):
    before = batch["before_grid"].to(dev)
    after = batch["after_grid"].to(dev)
    if variant["family"] in ("repo_model",):
        out = model(torch.cat([before, after], dim=1))
    else:
        out = model(before_grid=before, after_grid=after)
    logits = out.get("cause_logits", out.get("remap_logits"))
    return logits, out

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
            logits, out = forward_logits(model, batch, dev)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            if variant["family"] == "evidence" and cfg["primitive_mask_weight"] > 0:
                masks = batch["primitive_masks"].to(dev)
                bce = F.binary_cross_entropy(out["evidence_maps"], masks)
                inter = (out["evidence_maps"] * masks).sum(dim=(2, 3))
                denom = out["evidence_maps"].sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
                dice = (1.0 - (2.0 * inter + 1e-5) / (denom + 1e-5)).mean()
                loss = loss + cfg["primitive_mask_weight"] * (bce + dice)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * target.shape[0]
            n += target.shape[0]
        vals, tgts = [], []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                logits, _ = forward_logits(model, batch, dev)
                vals.append(torch.sigmoid(logits).cpu().numpy())
                tgts.append(batch["target_multihot"].numpy())
        val = _metrics_at_threshold(np.concatenate(vals), np.concatenate(tgts), 0.5)
        if val["exact_match"] > best_exact:
            best_exact = val["exact_match"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        row = {"epoch": epoch, "train_loss": total / max(1, n), "val_exact": val["exact_match"], "val_macro_f1": val["macro_f1"], "seconds": time.time() - started}
        log_rows.append(row)
        print(json.dumps({"run": run_name, **row}), flush=True)
    for row in log_rows:
        with (run_dir / "train_log.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
    ckpt = {"model_state": best_state, "config": cfg, "config_hash": config_hash(cfg), "model_key": model_key, "seed": seed, "fold_id": fold["fold_id"]}
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
            logits, out = forward_logits(model, batch, dev)
            prob = torch.sigmoid(logits)
            values["pred"].append(prob.cpu().numpy())
            values["logits"].append(logits.cpu().numpy())
            values["target"].append(batch["target_multihot"].numpy())
            if "evidence_maps" in out:
                maps = out["evidence_maps"].cpu().numpy()
            else:
                maps = np.zeros((len(batch["target_multihot"]), 4, batch["before_grid"].shape[2], batch["before_grid"].shape[3]), dtype=np.float32)
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

def eval_model(model):
    val_values = collect_generic(model, cfg["val_split"])
    val_f1 = _tune_thresholds(val_values["pred"], val_values["target"])
    val_exact = _tune_thresholds_exact(val_values["pred"], val_values["target"])
    val_evidence = _tune_evidence_threshold(val_values["masks"], val_values["target"], topk=3)
    splits = {}
    for split in ("test_single", "test_composed_seen", "test_composed_heldout"):
        splits[split] = evaluate_collected(
            collect_generic(model, split),
            val_f1_thresholds=val_f1,
            val_exact_thresholds=val_exact,
            val_evidence_threshold=val_evidence,
        )
    primitive = {"test_composed_heldout": primitive_diff_rule(split_path(str(data_dir), "test_composed_heldout"))}
    return splits, primitive, {
        "source_split": cfg["val_split"],
        "val_f1_tuned_thresholds": val_f1,
        "val_exact_tuned_thresholds": val_exact,
        "val_evidence_threshold_topk3": val_evidence,
        "primary_metric": "exact_fixed_0p5",
        "secondary_metric": "exact_val_exact_tuned",
    }

def diagnostic_eval():
    split = "test_composed_heldout"
    ds = CounterfactualTransitionDataset(split_path(str(data_dir), split))
    preds, targets, masks, mask_targets, roles, inactive = [], [], [], [], [], []
    for item in ds:
        target = item["target_multihot"].numpy()
        pm = item["primitive_masks"].numpy()
        if variant["family"] in ("primitive_rule", "true_mask"):
            pred = (pm.reshape(4, -1).sum(axis=1) > 0).astype(np.float32)
        else:
            pred = target.astype(np.float32)
        preds.append(pred); targets.append(target); masks.append(pm); mask_targets.append(pm)
        roles.append(int(item["transition_role_id"]))
        inactive.extend(pm.mean(axis=(1, 2))[target <= 0.5].tolist())
    values = {
        "pred": np.stack(preds), "logits": np.stack(preds) * 20 - 10,
        "target": np.stack(targets), "masks": np.stack(masks),
        "mask_targets": np.stack(mask_targets), "roles": np.array(roles),
        "inactive": inactive,
    }
    metrics = evaluate_collected(values, val_f1_thresholds=[0.5]*4, val_exact_thresholds=[0.5]*4, val_evidence_threshold=0.5)
    return {"test_composed_heldout": metrics}, {"test_composed_heldout": primitive_diff_rule(split_path(str(data_dir), split))}, {}

started = time.time()
if variant["family"] in ("primitive_rule", "oracle_tuple", "true_mask"):
    splits, primitive, thresholds = diagnostic_eval()
    ckpt = None
else:
    model = train_model()
    splits, primitive, thresholds = eval_model(model)
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
    "checkpoint_used": ckpt,
    "config_hash": config_hash(cfg),
    "commit": commit_sha(),
    "modal_resource": {"gpu": None, "function": "run_one"},
    "runtime_seconds": time.time() - started,
    "notes": variant.get("notes", ""),
}
eval_name = f"eval_{model_key}_{fold_name}_seed{seed}.json"
write_json(run_dir / eval_name, report)
write_json(run_dir / "eval_counterfactual_difference.json", report)
if splits["test_composed_heldout"].get("faithfulness") is not None:
    write_json(run_dir / "faithfulness_audit.json", splits["test_composed_heldout"].get("faithfulness"))
manifest = {k: report[k] for k in ("run_name", "model_key", "display_name", "fold_id", "fold_name", "seed", "commit", "config_hash", "modal_resource", "runtime_seconds")}
manifest["command"] = "modal run scripts/modal_run_final_evidence_map.py"
write_json(run_dir / "manifest.json", manifest)
print(json.dumps({"done": run_name, "exact": splits["test_composed_heldout"].get("exact_fixed_0p5")}), flush=True)
'''
    env = os.environ.copy()
    env["FINAL_PAYLOAD"] = json.dumps(payload)
    subprocess.check_call(["python3", "-c", code], env=env)
    output_volume.commit()
    return payload


@app.function(image=image, volumes={"/out": output_volume}, timeout=30 * 60)
def aggregate():
    _remote_setup()
    subprocess.check_call(["python3", "scripts/aggregate_final_evidence_map_modal.py", "--root", str(OUT_ROOT)])
    output_volume.commit()
    return str(OUT_ROOT)


def _payloads(include_seed2=True):
    seeds = SEEDS if include_seed2 else (0, 1)
    payloads = []
    for fold in FOLDS:
        for seed in seeds:
            for variant in TRAIN_VARIANTS + DIAGNOSTIC_VARIANTS:
                payloads.append({"fold": fold, "seed": seed, "variant": variant.__dict__})
    return payloads


def _launch(include_seed2=True):
    if not PROFILE:
        raise RuntimeError("Set MODAL_PROFILE explicitly before submission")
    print(f"Modal profile before submission: {PROFILE}")
    print(inspect_data.remote())
    jobs = _payloads(include_seed2=include_seed2)
    print(f"Submitting {len(jobs)} final evidence-map jobs")
    completed = list(run_one.map(jobs, return_exceptions=False))
    print(f"Completed {len(completed)} jobs")
    print(aggregate.remote())


@app.local_entrypoint()
def launch_full():
    _launch(include_seed2=True)


@app.local_entrypoint()
def launch_seed01():
    _launch(include_seed2=False)
