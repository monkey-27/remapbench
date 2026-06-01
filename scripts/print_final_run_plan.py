"""Print the exact final training and evaluation plan without submitting jobs."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.make_final_configs import configs


def _checkpoint(cfg):
    if cfg["model"] == "cpo":
        return f"results/{cfg['run_name']}/best_val_tuple_loss.pt"
    return f"results/{cfg['run_name']}/best_val_composed_seen_exact.pt"


def _group(cfg):
    if cfg["model"] == "cpo":
        return "Account 5: tuple oracle jobs"
    if "seen_upper_bound" in cfg["run_name"]:
        return "Account 6: seen-pair upper-bound jobs"
    return f"Account {int(cfg['fold_id']) + 1}: fold {cfg['fold_id']} direct jobs"


def main():
    rows = configs()
    for index, (path, cfg) in enumerate(rows, 1):
        checkpoint = _checkpoint(cfg)
        if cfg["model"] == "cpo":
            train = f"python3 scripts/train_final_tuple.py --config {path}"
            evaluate = (
                f"python3 scripts/evaluate_oracle_tuple_gap.py --config {path} "
                f"--checkpoint {checkpoint}")
            expected = f"results/{cfg['run_name']}/oracle_tuple_gap.json"
        else:
            train = f"python3 scripts/train_final_direct.py --config {path}"
            evaluate = (
                f"python3 scripts/evaluate_final.py --config {path} --checkpoint {checkpoint}")
            expected = f"results/{cfg['run_name']}/eval_final.json"
            if cfg["model"] in ("gated_erpm", "factorized_gates") and "seen_upper_bound" not in cfg["run_name"]:
                evaluate += (
                    f"\n  python3 scripts/evaluate_causal_patching.py --config {path} "
                    f"--checkpoint {checkpoint}")
                expected += f", results/{cfg['run_name']}/causal_patching.json"
        print(f"job_id: J{index:02d}")
        print(f"account_group: {_group(cfg)}")
        print(f"config: {path}")
        print(f"model: {cfg['report_model_name']} ({cfg['model']})")
        print(f"fold: {cfg.get('fold_name')}")
        print(f"seed: {cfg['seed']}")
        print(f"train: {train}")
        print(f"evaluate: {evaluate}")
        print(f"expected: {expected}")
        print()
    print("Account 7: evaluation, patching, dataset evidence audit, aggregation, reserve")
    print("python3 scripts/final_dataset_evidence_audit.py")
    print("python3 scripts/aggregate_final_results.py")
    print(f"Total training jobs: {len(rows)}")


if __name__ == "__main__":
    main()
