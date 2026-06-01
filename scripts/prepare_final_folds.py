"""Generate, audit, and manifest the four final folds or a tiny smoke fold."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from remapbench.generate import generate_dataset
from remapbench.interventions import COMPOSED_PAIRS
from remapbench.validate import validate
from scripts.audit_dataset import audit as audit_dataset
from scripts.audit_tuples import audit as audit_tuples
from scripts.final_common import FOLDS, config_hash


FULL_COUNTS = {
    "n_train": 12000, "n_val": 2000, "n_test": 2000, "n_composed": 2000,
    "n_test_larger": 1000, "n_test_noisy": 1000,
    "n_train_composed_seen": 4000, "n_val_composed_seen": 1000,
    "n_test_composed_seen": 1000, "n_test_composed_heldout": 1000,
    "n_train_tuple_seen": 3000, "n_val_tuple_seen": 500,
    "n_test_tuple_seen": 500, "n_test_tuple_heldout": 500,
}
SMOKE_COUNTS = {
    "n_train": 80, "n_val": 40, "n_test": 40, "n_composed": 16,
    "n_test_larger": 40, "n_test_noisy": 40,
    "n_train_composed_seen": 24, "n_val_composed_seen": 12,
    "n_test_composed_seen": 12, "n_test_composed_heldout": 12,
    "n_train_tuple_seen": 8, "n_val_tuple_seen": 4,
    "n_test_tuple_seen": 4, "n_test_tuple_heldout": 4,
}


def _saved_sizes(root):
    metadata = json.loads((root / "metadata.json").read_text())
    return metadata["requested_split_sizes"], metadata["split_sizes"]


def _run_fold(root, fold, counts, seed=0, include_all=False, require_tuples=True):
    root.mkdir(parents=True, exist_ok=True)
    pair_id = fold["heldout_pair_id"]
    config = {
        "seed": seed,
        "heldout_composed_pair_id": pair_id,
        "include_all_composed_pairs_train": include_all,
        **counts,
    }
    generate_dataset(
        out_dir=str(root), seed=seed, heldout_composed_pair_id=pair_id,
        include_all_composed_pairs_train=include_all, **counts)
    valid = bool(validate(str(root)))
    dataset_report = audit_dataset(str(root))
    tuple_report = audit_tuples(str(root)) if require_tuples else None
    requested, saved = _saved_sizes(root)
    manifest = {
        "fold_id": fold["fold_id"],
        "fold_name": fold["fold_name"],
        "heldout_pair_id": pair_id,
        "heldout_pair_names": list(COMPOSED_PAIRS[pair_id]),
        "config_hash": config_hash(config),
        "dataset_path": str(root),
        "requested_split_sizes": requested,
        "saved_split_sizes": saved,
        "validate_ok": valid,
        "strict_dataset_audit_ok": dataset_report["pilot_ready"],
        "strict_tuple_audit_ok": tuple_report["ok"] if tuple_report else None,
        "audit_status": (
            "ready" if valid and dataset_report["pilot_ready"]
            and (tuple_report is None or tuple_report["ok"]) else "failed"),
    }
    (root / "fold_manifest.json").write_text(json.dumps(manifest, indent=2))
    (root / "audit_report.json").write_text(json.dumps(dataset_report, indent=2))
    if tuple_report:
        (root / "tuple_audit_report.json").write_text(json.dumps(tuple_report, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fold-id", type=int, choices=range(len(FOLDS)))
    parser.add_argument("--seen-upper-bound-only", action="store_true")
    parser.add_argument("--out", default="data/final_decomp_gap")
    args = parser.parse_args()
    if args.fold_id is not None and args.seen_upper_bound_only:
        parser.error("--fold-id and --seen-upper-bound-only are mutually exclusive")
    counts = SMOKE_COUNTS if args.smoke else FULL_COUNTS
    folds = [FOLDS[args.fold_id]] if args.fold_id is not None else (
        [FOLDS[2]] if args.smoke else ([] if args.seen_upper_bound_only else FOLDS))
    for fold in folds:
        root = Path(args.out) if args.smoke else Path(args.out) / fold["fold_name"]
        action = "would generate/audit" if args.dry_run else "generating/auditing"
        print(f"{action}: {fold['fold_name']} pair={fold['heldout_pair_id']} "
              f"{COMPOSED_PAIRS[fold['heldout_pair_id']]} -> {root}")
        print(json.dumps(counts, sort_keys=True))
        if not args.dry_run:
            manifest = _run_fold(root, fold, counts)
            print(json.dumps(manifest, indent=2))
    if not args.smoke and args.fold_id is None:
        root = Path(args.out) / "seen_upper_bound"
        action = "would generate/audit" if args.dry_run else "generating/auditing"
        print(f"{action}: seen_upper_bound include_all_composed_pairs_train=True -> {root}")
        seen_counts = {**counts, "n_train_tuple_seen": 0, "n_val_tuple_seen": 0,
                       "n_test_tuple_seen": 0, "n_test_tuple_heldout": 0}
        print(json.dumps(seen_counts, sort_keys=True))
        if not args.dry_run:
            _run_fold(root, {**FOLDS[2], "fold_id": "seen_upper_bound",
                             "fold_name": "seen_upper_bound"}, seen_counts,
                      include_all=True, require_tuples=False)


if __name__ == "__main__":
    main()
