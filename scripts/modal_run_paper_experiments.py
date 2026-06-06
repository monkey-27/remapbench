"""Modal final paper experiment suite for decomposition-gap results."""
import json
import os
import pathlib
import shutil
import subprocess
from dataclasses import dataclass

import modal


ROOT = pathlib.Path(__file__).resolve().parents[1]
REMOTE = pathlib.Path("/root/iconip_monkey")
DATA_VOLUME_NAME = "iconip-monkey-final-decomp-gap-data"
OUTPUT_VOLUME_NAME = "iconip-monkey-final-evidence-map-results"
OUT_ROOT = pathlib.Path("results/paper_final")
PROFILE = os.environ.get("MODAL_PROFILE", "")
if PROFILE == "chat-arjunc":
    raise RuntimeError("Refusing to use disallowed Modal profile: chat-arjunc")

app = modal.App("iconip-monkey-paper-final")
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
    "test_composed", "test_larger", "test_noisy",
    "train_composed_seen", "val_composed_seen",
    "test_composed_seen", "test_composed_heldout",
    "train_tuple_seen", "val_tuple_seen", "test_tuple_seen",
    "test_tuple_heldout",
)


@dataclass(frozen=True)
class Variant:
    key: str
    display: str
    family: str
    model_name: str | None = None
    readout: str | None = None
    primitive_mask_weight: float = 0.0
    mask_loss_balancing: str = "none"
    active_cause_weight: float = 1.0
    inactive_cause_weight: float = 1.0
    positive_pixel_weight: float = 1.0
    use_diff_channels: bool = True
    train_splits: tuple[str, ...] = ("train_single", "train_composed_seen")
    notes: str = ""
    epochs: int = 15


VARIANTS = [
    Variant("cnn_diff", "CNN + diff", "direct_cnn",
            notes="Direct CNN over before/after/diff/absdiff."),
    Variant("grid_token_transformer", "Grid-token Transformer", "transformer",
            notes="Compact transformer over grid-cell tokens."),
    Variant("cell_graph_gnn", "Cell-graph GNN", "gnn",
            notes="Compact 4-neighbor message passing baseline."),
    Variant("factorized_gate_head", "Factorized gate-head", "repo_model", model_name="factorized_gates",
            notes="Naive independent/factorized gate heads over shared features."),
    Variant("primitive_diff_rule", "Primitive diff rule", "primitive_rule", epochs=0,
            notes="Diagnostic primitive-channel diff rule, not an oracle."),
    Variant("seen_pair_control", "Seen-pair control", "direct_cnn",
            train_splits=("train_single", "train_composed_seen", "test_composed_heldout"),
            notes="Privileged learnability control with heldout pair included in training."),
    Variant("oracle_tuple_composer", "Oracle tuple composer", "oracle_tuple", epochs=0,
            notes="Privileged upper bound once decomposition is supplied."),
    Variant("evidence_mask_learned_classifier", "Evidence-mask + learned classifier", "evidence",
            readout="learned_classifier", primitive_mask_weight=1.0,
            notes="Primitive evidence supervision with learned hidden-feature classifier."),
    Variant("evidence_map_noisy_or_unbalanced", "Evidence-map noisy-or, unbalanced", "evidence",
            readout="evidence_noisy_or", primitive_mask_weight=1.0,
            notes="Faithful noisy-or evidence-map readout with unbalanced primitive mask loss."),
    Variant("evidence_map_noisy_or_active_balanced", "Evidence-map noisy-or, active-balanced", "evidence",
            readout="evidence_noisy_or", primitive_mask_weight=1.0,
            mask_loss_balancing="active_cause_balanced", inactive_cause_weight=0.25,
            notes="Final method: noisy-or readout and active-balanced primitive evidence supervision."),
    Variant("bce_only", "BCE only", "evidence", readout="learned_classifier",
            primitive_mask_weight=0.0,
            notes="Labels only; no primitive mask supervision."),
    Variant("evidence_map_noisy_or_active_balanced_no_diff", "Active-balanced noisy-or, no diff", "evidence",
            readout="evidence_noisy_or", primitive_mask_weight=1.0,
            mask_loss_balancing="active_cause_balanced", inactive_cause_weight=0.25,
            use_diff_channels=False,
            notes="Optional final-method input ablation without signed/absolute diff channels."),
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
    (local_results / "paper_final" / "runs").mkdir(parents=True, exist_ok=True)
    (local_results / "paper_final" / "prereq").mkdir(parents=True, exist_ok=True)
    (local_results / "paper_final" / "visualizations").mkdir(parents=True, exist_ok=True)


def _run_python(code, env=None):
    subprocess.check_call(["python3", "-c", code], env=env)


@app.function(image=image, volumes={"/data": data_volume, "/out": output_volume}, timeout=20 * 60)
def inspect_data():
    _remote_setup()
    code = r'''
import json
from collections import Counter
from pathlib import Path
import numpy as np
from remapbench.interventions import COMPOSED_PAIRS

folds = [
    {"fold_id": 0, "fold_name": "fold0_goal_topology", "heldout_pair_id": 0},
    {"fold_id": 1, "fold_name": "fold1_sensory_action", "heldout_pair_id": 1},
    {"fold_id": 2, "fold_name": "fold2_goal_action", "heldout_pair_id": 2},
    {"fold_id": 3, "fold_name": "fold3_sensory_topology", "heldout_pair_id": 3},
]
splits = ["train_single", "val_single", "test_single", "test_composed", "test_larger", "test_noisy", "train_composed_seen", "val_composed_seen", "test_composed_seen", "test_composed_heldout", "train_tuple_seen", "val_tuple_seen", "test_tuple_seen", "test_tuple_heldout"]
candidates = [Path("/data/final_decomp_gap"), Path("/data"), Path("/data/data/final_decomp_gap")]
root = next((p for p in candidates if all((p / f["fold_name"]).is_dir() for f in folds)), None)
if root is None:
    raise SystemExit("Could not find clean Modal fold directories under " + ", ".join(map(str, candidates)))
report = {"status": "pass", "data_root": str(root), "folds": {}, "errors": [], "warnings": []}
for fold in folds:
    fold_name = fold["fold_name"]
    heldout = int(fold["heldout_pair_id"])
    fold_dir = root / fold_name
    fold_report = {
        "fold_id": fold["fold_id"],
        "heldout_pair_id": heldout,
        "heldout_pair_names": list(COMPOSED_PAIRS[heldout]),
        "splits": {},
    }
    for split in splits:
        path = fold_dir / f"{split}.npz"
        row = {"exists": path.exists(), "path": str(path)}
        if not path.exists():
            report["errors"].append(f"{path}: missing")
        else:
            with np.load(path, allow_pickle=True) as data:
                row["n"] = int(len(data["target_multihot"]))
                row["label_counts"] = np.asarray(data["target_multihot"]).sum(axis=0).astype(float).tolist()
                if "intervention_pair_id" in data:
                    row["intervention_pair_id_counts"] = {str(k): int(v) for k, v in Counter(map(int, data["intervention_pair_id"])).items()}
                if "pair_id" in data:
                    row["pair_id_counts"] = {str(k): int(v) for k, v in Counter(map(int, data["pair_id"])).items()}
                if "tuple_pair_id" in data:
                    row["tuple_pair_id_counts"] = {str(k): int(v) for k, v in Counter(map(int, data["tuple_pair_id"])).items()}
                pair_values = set()
                if "intervention_pair_id" in data:
                    pair_values.update(int(x) for x in data["intervention_pair_id"] if int(x) >= 0)
                if "pair_id" in data:
                    pair_values.update(int(x) for x in data["pair_id"] if int(x) >= 0)
                if "tuple_pair_id" in data:
                    pair_values.update(int(x) for x in data["tuple_pair_id"] if int(x) >= 0)
                if split in ("train_composed_seen", "val_composed_seen", "test_composed_seen", "train_tuple_seen", "val_tuple_seen", "test_tuple_seen"):
                    if heldout in pair_values:
                        report["errors"].append(f"{path}: seen split leaks heldout pair {heldout}")
                if split in ("test_composed_heldout", "test_tuple_heldout"):
                    if pair_values and pair_values != {heldout}:
                        report["errors"].append(f"{path}: heldout split pair ids {sorted(pair_values)} != {heldout}")
        fold_report["splits"][split] = row
    report["folds"][fold_name] = fold_report
if report["errors"]:
    report["status"] = "fail"
out = Path("results/paper_final/prereq/data_integrity_report.json")
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
if report["status"] != "pass":
    raise SystemExit("Data integrity preflight failed")
'''
    _run_python(code)
    output_volume.commit()
    return "results/paper_final/prereq/data_integrity_report.json"


@app.function(image=image, volumes={"/data": data_volume, "/out": output_volume}, timeout=30 * 60)
def audit_all_folds():
    _remote_setup()
    code = r'''
import json, subprocess
from pathlib import Path
import yaml

folds = [
    {"fold_id": 0, "fold_name": "fold0_goal_topology", "heldout_pair_id": 0},
    {"fold_id": 1, "fold_name": "fold1_sensory_action", "heldout_pair_id": 1},
    {"fold_id": 2, "fold_name": "fold2_goal_action", "heldout_pair_id": 2},
    {"fold_id": 3, "fold_name": "fold3_sensory_topology", "heldout_pair_id": 3},
]
candidates = [Path("/data/final_decomp_gap"), Path("/data"), Path("/data/data/final_decomp_gap")]
root = next((p for p in candidates if all((p / f["fold_name"]).is_dir() for f in folds)), None)
if root is None:
    raise SystemExit("Could not find Modal data root")
out_root = Path("results/paper_final/prereq")
summary = {"status": "pass", "folds": {}, "errors": [], "warnings": []}
for fold in folds:
    cfg = {
        "run_name": f"paper_preflight_{fold['fold_name']}",
        "data_dir": str(root / fold["fold_name"]),
        "heldout_composed_pair_id": fold["heldout_pair_id"],
        "fold_id": fold["fold_id"],
        "fold_name": fold["fold_name"],
    }
    cfg_path = out_root / f"audit_config_{fold['fold_name']}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    out_path = out_root / f"audit_{fold['fold_name']}.json"
    subprocess.check_call(["python3", "scripts/audit_evidence_mask.py", "--config", str(cfg_path), "--out", str(out_path)])
    row = json.loads(out_path.read_text())
    summary["folds"][fold["fold_name"]] = {
        "status": row.get("status"),
        "errors": row.get("errors", []),
        "warnings": row.get("warnings", []),
        "label_balance": row.get("label_balance", {}),
    }
    summary["errors"].extend(row.get("errors", []))
    summary["warnings"].extend(row.get("warnings", []))
if summary["errors"]:
    summary["status"] = "fail"
summary["warnings"] = sorted(set(summary["warnings"]))
(out_root / "audit_preflight_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
if summary["status"] != "pass":
    raise SystemExit("Audit preflight failed")
'''
    _run_python(code)
    output_volume.commit()
    return "results/paper_final/prereq/audit_preflight_summary.json"


@app.function(image=image, volumes={"/out": output_volume}, timeout=10 * 60)
def write_expected_manifest(payloads):
    _remote_setup()
    rows = []
    for payload in payloads:
        fold = payload["fold"]
        variant = payload["variant"]
        seed = payload["seed"]
        run_name = f"paper_final_{variant['key']}_{fold['fold_name']}_seed{seed}"
        rows.append({
            "run_name": run_name,
            "model_key": variant["key"],
            "fold_id": fold["fold_id"],
            "fold_name": fold["fold_name"],
            "seed": seed,
            "source_branch": payload.get("source_branch"),
            "source_commit": payload.get("source_commit"),
            "eval_path": f"results/paper_final/runs/{run_name}/eval_{variant['key']}_{fold['fold_name']}_seed{seed}.json",
            "config_path": f"results/paper_final/runs/{run_name}/config.yaml",
            "audit_path": f"results/paper_final/runs/{run_name}/audit_counterfactual_difference.json",
            "manifest_path": f"results/paper_final/runs/{run_name}/manifest.json",
        })
    report = {
        "status": "complete",
        "expected_runs": len(rows),
        "data_volume": "iconip-monkey-final-decomp-gap-data",
        "results_volume": "iconip-monkey-final-evidence-map-results",
        "runs": rows,
    }
    path = pathlib.Path("results/paper_final/prereq/expected_run_manifest.json")
    path.write_text(json.dumps(report, indent=2))
    output_volume.commit()
    return str(path)


@app.function(image=image, volumes={"/data": data_volume, "/out": output_volume}, timeout=3 * 60 * 60)
def run_one(payload):
    _remote_setup()
    code = r'''
import json, os, subprocess, sys, time
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
from remapbench.counterfactual_difference import CounterfactualTransitionDataset, split_path
from scripts.evaluate_counterfactual_difference import (
    _metrics_at_threshold, _tune_thresholds, _tune_thresholds_exact,
    _tune_evidence_threshold, evaluate_collected, primitive_diff_rule,
)
from scripts.final_common import config_hash, commit_sha, write_json

payload = json.loads(os.environ["PAPER_PAYLOAD"])
fold = payload["fold"]
variant = payload["variant"]
seed = int(payload["seed"])
fold_name = fold["fold_name"]
model_key = variant["key"]
run_name = f"paper_final_{model_key}_{fold_name}_seed{seed}"
root = Path("results/paper_final")
run_dir = root / "runs" / run_name
eval_name = f"eval_{model_key}_{fold_name}_seed{seed}.json"
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
    "mask_loss_balancing": variant.get("mask_loss_balancing", "none"),
    "active_cause_weight": float(variant.get("active_cause_weight", 1.0)),
    "inactive_cause_weight": float(variant.get("inactive_cause_weight", 1.0)),
    "positive_pixel_weight": float(variant.get("positive_pixel_weight", 1.0)),
    "inactive_weight": 0.0,
    "context_difference_weight": 0.0,
    "mixed_union_weight": 0.0,
    "map_context_weight": 0.0,
    "map_union_weight": 0.0,
    "use_diff_channels": bool(variant.get("use_diff_channels", True)),
}
(run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
if (run_dir / eval_name).exists():
    existing = json.loads((run_dir / eval_name).read_text())
    existing_manifest = existing.get("manifest", {})
    if (
        existing.get("status") == "complete"
        and existing.get("config_hash") == config_hash(cfg)
        and (existing.get("commit") in (None, payload.get("source_commit")) or payload.get("source_commit") is None)
        and existing_manifest.get("data_path") == str(data_dir)
        and existing.get("audit", {}).get("status") == "pass"
    ):
        print(json.dumps({"skip_complete": run_name, "config_hash": config_hash(cfg)}), flush=True)
        raise SystemExit(0)
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
    if variant["family"] == "repo_model":
        out = model(torch.cat([before, after], dim=1))
    else:
        out = model(before_grid=before, after_grid=after)
    logits = out.get("cause_logits", out.get("remap_logits"))
    return logits, out

def primitive_mask_loss(evidence_maps, masks, target):
    bce = F.binary_cross_entropy(evidence_maps, masks, reduction="none").mean(dim=(2, 3))
    inter = (evidence_maps * masks).sum(dim=(2, 3))
    denom = evidence_maps.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
    dice = 1.0 - (2.0 * inter + 1e-5) / (denom + 1e-5)
    per_cause = bce + dice
    if cfg["mask_loss_balancing"] == "active_cause_balanced":
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
            logits, out = forward_logits(model, batch, dev)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            if variant["family"] == "evidence" and cfg["primitive_mask_weight"] > 0:
                masks = batch["primitive_masks"].to(dev)
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
                logits, _ = forward_logits(model, batch, dev)
                vals.append(torch.sigmoid(logits).cpu().numpy())
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
    values = {
        "pred": [], "logits": [], "target": [], "masks": [],
        "mask_targets": [], "before": [], "after": [], "roles": [],
        "inactive": [],
    }
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
            values["before"].append(batch["before_grid"].numpy())
            values["after"].append(batch["after_grid"].numpy())
            values["roles"].extend(batch["transition_role_id"].numpy().tolist())
            inactive = batch["target_multihot"].numpy() <= 0.5
            values["inactive"].extend(maps.mean(axis=(2, 3))[inactive].tolist())
    return {
        "pred": np.concatenate(values["pred"]),
        "logits": np.concatenate(values["logits"]),
        "target": np.concatenate(values["target"]),
        "masks": np.concatenate(values["masks"]),
        "mask_targets": np.concatenate(values["mask_targets"]),
        "before": np.concatenate(values["before"]),
        "after": np.concatenate(values["after"]),
        "roles": np.array(values["roles"]),
        "inactive": values["inactive"],
    }

def fold3_rescue_diagnostics(values):
    target = values["target"]
    pred = values["pred"]
    maps = values["masks"]
    mask_targets = values["mask_targets"]
    active_topology = target[:, 2] > 0.5
    topo = maps[active_topology, 2]
    topo_target = mask_targets[active_topology, 2]
    flat = topo.reshape(topo.shape[0], -1) if topo.size else np.zeros((0, 1), dtype=np.float32)
    pred_bin = topo > 0.5
    target_bin = topo_target > 0.5
    inter = (pred_bin & target_bin).sum(axis=(1, 2)) if len(topo) else np.array([])
    denom = pred_bin.sum(axis=(1, 2)) + target_bin.sum(axis=(1, 2)) if len(topo) else np.array([])
    dice = (2 * inter + 1e-5) / (denom + 1e-5) if len(topo) else np.array([])
    if len(flat):
        top3 = np.sort(flat, axis=1)[:, -3:].mean(axis=1)
        max_mean = float(flat.max(axis=1).mean())
        top3_mean = float(top3.mean())
        area_mean = float((flat > 0.5).mean(axis=1).mean())
    else:
        max_mean = top3_mean = area_mean = None
    return {
        "topology_evidence_max_mean": max_mean,
        "topology_evidence_top3_mean": top3_mean,
        "topology_evidence_area_gt_0p5_mean": area_mean,
        "topology_active_dice": float(dice.mean()) if len(dice) else None,
        "value_fp_count": int(((pred[:, 1] > 0.5) & (target[:, 1] <= 0.5)).sum()),
        "action_fp_count": int(((pred[:, 3] > 0.5) & (target[:, 3] <= 0.5)).sum()),
    }

def save_fold3_visualizations(values):
    if fold["fold_id"] != 3 or variant["family"] != "evidence":
        return None
    out_dir = root / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = values["target"]
    pred = values["pred"]
    wanted = np.where(target[:, 2] > 0.5)[0][:8]
    if len(wanted) == 0:
        return None
    path = out_dir / f"fold3_examples_{model_key}_seed{seed}.npz"
    np.savez_compressed(
        path,
        indices=wanted,
        target=target[wanted],
        pred=pred[wanted],
        before_grid=values["before"][wanted],
        after_grid=values["after"][wanted],
        evidence_maps=values["masks"][wanted],
        primitive_masks=values["mask_targets"][wanted],
    )
    return str(path)

def eval_model(model):
    val_values = collect_generic(model, cfg["val_split"])
    val_f1 = _tune_thresholds(val_values["pred"], val_values["target"])
    val_exact = _tune_thresholds_exact(val_values["pred"], val_values["target"])
    val_evidence = _tune_evidence_threshold(val_values["masks"], val_values["target"], topk=3)
    splits = {}
    heldout_values = None
    for split in ("test_single", "test_composed_seen", "test_composed_heldout"):
        values = collect_generic(model, split)
        if split == "test_composed_heldout":
            heldout_values = values
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
    return splits, primitive, thresholds, fold3_rescue_diagnostics(heldout_values), save_fold3_visualizations(heldout_values)

def diagnostic_eval():
    split = "test_composed_heldout"
    ds = CounterfactualTransitionDataset(split_path(str(data_dir), split))
    preds, targets, masks, mask_targets, roles, inactive = [], [], [], [], [], []
    if variant["family"] == "oracle_tuple":
        from remapbench.counterfactual_difference import CounterfactualTupleDataset
        ds = CounterfactualTupleDataset(split_path(str(data_dir), "test_tuple_heldout"))
        for item in ds:
            a = item["base_to_A"]["target_multihot"].numpy()
            b = item["base_to_B"]["target_multihot"].numpy()
            ab = item["base_to_AB"]
            target = ab["target_multihot"].numpy()
            pred = np.clip(a + b, 0, 1).astype(np.float32)
            pm = ab["primitive_masks"].numpy()
            preds.append(pred); targets.append(target); masks.append(pm); mask_targets.append(pm)
            roles.append(int(ab["transition_role_id"]))
            inactive.extend(pm.mean(axis=(1, 2))[target <= 0.5].tolist())
    elif variant["family"] == "primitive_rule":
        for item in ds:
            target = item["target_multihot"].numpy()
            pm = item["primitive_masks"].numpy()
            pred = (pm.reshape(4, -1).sum(axis=1) > 0).astype(np.float32)
            preds.append(pred); targets.append(target); masks.append(pm); mask_targets.append(pm)
            roles.append(int(item["transition_role_id"]))
            inactive.extend(pm.mean(axis=(1, 2))[target <= 0.5].tolist())
    else:
        raise ValueError(f"unsupported diagnostic family {variant['family']}")
    values = {
        "pred": np.stack(preds), "logits": np.stack(preds) * 20 - 10,
        "target": np.stack(targets), "masks": np.stack(masks),
        "mask_targets": np.stack(mask_targets), "roles": np.array(roles),
        "inactive": inactive,
    }
    metrics = evaluate_collected(values, val_f1_thresholds=[0.5]*4, val_exact_thresholds=[0.5]*4, val_evidence_threshold=0.5)
    return {"test_composed_heldout": metrics}, {"test_composed_heldout": primitive_diff_rule(split_path(str(data_dir), split))}, {}, fold3_rescue_diagnostics(values), None

def audit_payload():
    audit_path = run_dir / "audit_counterfactual_difference.json"
    audit = json.loads(audit_path.read_text())
    return {
        "passed": audit.get("status") == "pass",
        "status": audit.get("status"),
        "errors": audit.get("errors", []),
        "warnings": audit.get("warnings", []),
    }

started = time.time()
if variant["family"] in ("primitive_rule", "oracle_tuple"):
    splits, primitive, thresholds, fold3_diag, visualization_path = diagnostic_eval()
    ckpt = None
else:
    model = train_model()
    splits, primitive, thresholds, fold3_diag, visualization_path = eval_model(model)
    ckpt = str(run_dir / "best_val_composed_seen_exact.pt")
manifest = {
    "branch": payload.get("source_branch"),
    "commit": payload.get("source_commit") or commit_sha(),
    "model": model_key,
    "fold": fold,
    "seed": seed,
    "config_hash": config_hash(cfg),
    "data_path": str(data_dir),
    "command": "modal run scripts/modal_run_paper_experiments.py",
    "modal_resources": {"gpu": None, "function": "run_one"},
}
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
    "fold3_rescue_diagnostics": fold3_diag,
    "visualization_path": visualization_path,
    "audit": audit_payload(),
    "checkpoint_used": ckpt,
    "config_hash": config_hash(cfg),
    "commit": payload.get("source_commit") or commit_sha(),
    "manifest": manifest,
    "mask_loss_balancing": cfg["mask_loss_balancing"],
    "active_cause_weight": cfg["active_cause_weight"],
    "inactive_cause_weight": cfg["inactive_cause_weight"],
    "modal_resource": manifest["modal_resources"],
    "runtime_seconds": time.time() - started,
    "notes": variant.get("notes", ""),
}
manifest["runtime_seconds"] = report["runtime_seconds"]
manifest["audit_status"] = report["audit"]["status"]
write_json(run_dir / eval_name, report)
write_json(run_dir / "eval_counterfactual_difference.json", report)
write_json(run_dir / "manifest.json", manifest)
if splits["test_composed_heldout"].get("faithfulness") is not None:
    write_json(run_dir / "faithfulness_audit.json", splits["test_composed_heldout"].get("faithfulness"))
print(json.dumps({"done": run_name, "exact": splits["test_composed_heldout"].get("exact_fixed_0p5"), "audit": report["audit"]["status"]}), flush=True)
'''
    env = os.environ.copy()
    env["PAPER_PAYLOAD"] = json.dumps(payload)
    _run_python(code, env=env)
    output_volume.commit()
    return payload


@app.function(image=image, volumes={"/out": output_volume}, timeout=30 * 60)
def aggregate(allow_incomplete=False):
    _remote_setup()
    cmd = ["python3", "scripts/aggregate_paper_experiments.py", "--root", str(OUT_ROOT)]
    if allow_incomplete:
        cmd.append("--allow-incomplete")
    subprocess.check_call(cmd)
    output_volume.commit()
    return str(OUT_ROOT)


def _payloads():
    source = _source_info()
    payloads = []
    for fold in FOLDS:
        for seed in SEEDS:
            for variant in VARIANTS:
                payloads.append({"fold": fold, "seed": seed, "variant": variant.__dict__, **source})
    return payloads


def _git_text(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_info():
    return {
        "source_branch": _git_text("branch", "--show-current"),
        "source_commit": _git_text("rev-parse", "HEAD"),
    }


def _launch():
    if not PROFILE:
        raise RuntimeError("Set MODAL_PROFILE explicitly before submission")
    print(f"Modal profile before submission: {PROFILE}")
    print(inspect_data.remote())
    print(audit_all_folds.remote())
    jobs = _payloads()
    print(write_expected_manifest.remote(jobs))
    print(f"Submitting {len(jobs)} paper-final jobs")
    completed = list(run_one.map(jobs, return_exceptions=False))
    print(f"Completed {len(completed)} jobs")
    print(aggregate.remote(False))


@app.local_entrypoint()
def launch_full():
    _launch()


@app.local_entrypoint()
def aggregate_only(allow_incomplete: bool = False):
    if not PROFILE:
        raise RuntimeError("Set MODAL_PROFILE explicitly before aggregation")
    print(f"Modal profile before aggregation: {PROFILE}")
    print(aggregate.remote(allow_incomplete))
