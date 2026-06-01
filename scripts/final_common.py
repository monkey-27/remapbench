"""Shared metadata and metric helpers for the decomposition-gap final run."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml

from remapbench.interventions import COMPOSED_PAIRS


PATHWAY_NAMES = ("sensory", "value", "map", "action")
TARGET_NAMES = ("sensory_update", "value_remap", "map_remap", "action_remap")
FOLDS = [
    {"fold_id": 0, "fold_name": "fold0_goal_topology", "heldout_pair_id": 0},
    {"fold_id": 1, "fold_name": "fold1_sensory_action", "heldout_pair_id": 1},
    {"fold_id": 2, "fold_name": "fold2_goal_action", "heldout_pair_id": 2},
    {"fold_id": 3, "fold_name": "fold3_sensory_topology", "heldout_pair_id": 3},
]


def fold_for_id(fold_id):
    return next(fold for fold in FOLDS if fold["fold_id"] == int(fold_id))


def pair_names(pair_id):
    return list(COMPOSED_PAIRS[int(pair_id)])


def commit_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_dirty():
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def load_config(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def config_hash(config):
    payload = yaml.safe_dump(config, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def run_dir(config):
    return os.path.join("results", config["run_name"])


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path, payload):
    ensure_parent(path)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def common_metadata(config, config_path=None):
    pair_id = config.get("heldout_composed_pair_id")
    return {
        "commit": commit_sha(),
        "git_dirty": git_dirty(),
        "fold_id": config.get("fold_id"),
        "fold_name": config.get("fold_name"),
        "heldout_pair_id": pair_id,
        "heldout_pair_names": pair_names(pair_id) if pair_id is not None else None,
        "model": config.get("model"),
        "report_model_name": config.get("report_model_name", config.get("model")),
        "seed": config.get("seed"),
        "dataset_path": config.get("data_dir"),
        "config_path": config_path,
    }


def validate_checkpoint(config, checkpoint):
    expected = {
        "model_name": config.get("model"),
        "report_model_name": config.get("report_model_name", config.get("model")),
        "seed": config.get("seed"),
        "fold_id": config.get("fold_id"),
        "heldout_composed_pair_id": config.get("heldout_composed_pair_id"),
    }
    mismatches = {
        key: {"checkpoint": checkpoint.get(key), "config": value}
        for key, value in expected.items()
        if checkpoint.get(key) is not None and checkpoint.get(key) != value
    }
    if checkpoint.get("config_hash") not in (None, config_hash(config)):
        mismatches["config_hash"] = {
            "checkpoint": checkpoint.get("config_hash"), "config": config_hash(config)}
    if mismatches:
        raise ValueError(f"Checkpoint provenance mismatch: {mismatches}")


def stable_pathway_metrics(overall):
    per_label = overall["per_label"]
    metrics = {}
    for pathway, target in zip(PATHWAY_NAMES, TARGET_NAMES):
        row = per_label[target]
        positives = row["tp"] + row["fn"]
        negatives = overall["n_samples"] - positives
        metrics[f"miss_{pathway}"] = row["fn"] / positives if positives else None
        metrics[f"false_positive_{pathway}"] = row["fp"] / negatives if negatives else None
    planning = overall.get("planning", {})
    return {
        "exact_match": overall["exact_match"],
        "micro_f1": overall["micro_f1"],
        "macro_f1": overall["macro_f1"],
        "missed_remap_rates": overall["missed_remap_rates"],
        "false_structural_remap_on_sensory": overall["false_structural_remap_on_sensory"],
        "delta_future_mse": overall["delta_future_mse"],
        "delta_value_mse": overall["delta_value_mse"],
        "action_delta_mse": overall["action_delta_mse"],
        "planning_success_rate": planning.get("planning_success_rate"),
        "mean_step_regret": planning.get("mean_step_regret"),
        **metrics,
    }
