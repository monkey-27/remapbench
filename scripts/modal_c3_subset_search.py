"""Modal launcher for the 12-run C3 D1/D2 subset-search pilot."""
import os
import pathlib
import subprocess

import modal


ROOT = pathlib.Path(__file__).resolve().parents[1]
REMOTE = pathlib.Path("/root/iconip_monkey")
VOLUME_NAME = "iconip-monkey-c3-subset-search"
PROFILE = os.environ.get("MODAL_PROFILE", "")
if PROFILE == "chat-arjunc":
    raise RuntimeError("Refusing to use disallowed Modal profile: chat-arjunc")

app = modal.App("iconip-monkey-c3-subset-search")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml", "matplotlib", "torch")
    .add_local_dir(ROOT / "models", REMOTE / "models")
    .add_local_dir(ROOT / "remapbench", REMOTE / "remapbench")
    .add_local_dir(ROOT / "scripts", REMOTE / "scripts")
    .add_local_dir(ROOT / "configs", REMOTE / "configs")
    .add_local_dir(ROOT / "data" / "remapbench_v1_composed", REMOTE / "data" / "remapbench_v1_composed")
    .add_local_dir(ROOT / "data" / "remapbench_cpo_pilot", REMOTE / "data" / "remapbench_cpo_pilot")
)


def _prepare_output_links():
    for name in ("results", "figures"):
        target = pathlib.Path("/vol") / name
        target.mkdir(parents=True, exist_ok=True)
        link = REMOTE / name
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)


@app.function(image=image, gpu="A10G", volumes={"/vol": volume}, timeout=4 * 60 * 60)
def run_one(config_path, seed):
    os.chdir(REMOTE)
    _prepare_output_links()
    cfg_name = pathlib.Path(config_path).stem
    print(f"START config={cfg_name} seed={seed}")
    subprocess.check_call(["python3", "scripts/train_c3.py", "--config", config_path, "--seed", str(seed)])
    run_name = f"c3_{cfg_name}_seed{seed}"
    source = pathlib.Path("results") / run_name
    target = pathlib.Path("results/c3_subset_search_pilot/runs") / run_name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        import shutil
        shutil.rmtree(target)
    source.rename(target)
    subprocess.check_call([
        "python3", "scripts/evaluate_c3.py",
        "--config", config_path,
        "--checkpoint", str(target / "best_val_composed_seen_exact.pt"),
    ])
    volume.commit()
    print(f"DONE config={cfg_name} seed={seed}")
    return run_name


@app.function(image=image, volumes={"/vol": volume}, timeout=30 * 60)
def aggregate():
    os.chdir(REMOTE)
    _prepare_output_links()
    volume.reload()
    subprocess.check_call(["python3", "scripts/aggregate_c3.py"])
    volume.commit()
    return "results/c3_subset_search_pilot"


def _launch(configs):
    if not PROFILE:
        raise RuntimeError("Set MODAL_PROFILE explicitly before submission")
    print(f"Modal profile before submission: {PROFILE}")
    jobs = [(config, seed) for config in configs for seed in (0, 1)]
    completed = list(run_one.starmap(jobs))
    print(f"Completed {len(completed)} C3 runs")
    print(aggregate.remote())


@app.local_entrypoint()
def launch():
    _launch([
        f"configs/c3_subset_search/{dataset}_{variant}.yaml"
        for dataset in ("d1", "d2")
        for variant in ("full_reconstruction", "delta_space", "factor_scored")
    ])


@app.local_entrypoint()
def launch_sparsity_sweep():
    _launch([
        f"configs/c3_subset_search/{dataset}_{variant}{suffix}.yaml"
        for dataset in ("d1", "d2")
        for variant in ("full_reconstruction", "delta_space", "factor_scored")
        for suffix in ("_sparse0", "_sparse01")
    ])
