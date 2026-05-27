# Error-Gated Plasticity for Selective Remapping in Predictive Neural Maps

**RemapBench v0 — pilot codebase**

---

## Project Summary

Biological brains distinguish between changes that require updating an internal
predictive map (topology changes, reward shifts) and changes that are merely
perceptual noise (lighting, texture). This pilot implements **Error-Gated Plasticity
(ERPM)**: a mechanism that routes different error signals — sensory, value, future-state,
and action prediction errors — to distinct update pathways, enabling *selective* remapping.

### Mechanism

An ERPM agent maintains four internal pathways:

| Pathway | Updates when | Oracle label |
|---------|-------------|--------------|
| Sensory | Appearance/nuisance changes (no structural shift) | `sensory_update` |
| Value | Goal/reward location changes | `value_remap` |
| Predictive map | Topology / future-state structure changes | `map_remap` |
| Action | One-way / affordance constraints change | `action_remap` |

Each before→after gridworld pair is labelled with a multi-hot target
indicating which pathways should update.

---

## Quick Start

```bash
pip install -r requirements.txt
```

### Smoke commands (fast, run these first)

```bash
python scripts/generate_data.py  --config configs/smoke.yaml
python -m remapbench.validate    --data_dir data/smoke
python -m remapbench.visualize   --data_dir data/smoke --out_dir figures/smoke --n 4
python scripts/train.py          --config configs/smoke.yaml --model erpm --seed 0
python scripts/evaluate.py       --config configs/smoke.yaml \
    --checkpoint results/smoke_erpm/best.pt --split test_single
python scripts/evaluate.py       --config configs/smoke.yaml \
    --checkpoint results/smoke_erpm/best.pt --split test_composed
python scripts/make_plots.py     --config configs/smoke.yaml --run_dir results/smoke_erpm
```

### Pilot commands *(longer run; do not run by default)*

```bash
python scripts/generate_data.py  --config configs/pilot.yaml
python -m remapbench.validate    --data_dir data/remapbench_v0
python scripts/train.py          --config configs/pilot.yaml --model erpm --seed 0
python scripts/evaluate.py       --config configs/pilot.yaml \
    --checkpoint results/pilot_erpm_seed0/best.pt --split test_single
python scripts/evaluate.py       --config configs/pilot.yaml \
    --checkpoint results/pilot_erpm_seed0/best.pt --split test_composed
python scripts/make_plots.py     --config configs/pilot.yaml --run_dir results/pilot_erpm_seed0
```

---

## Package Layout

```
remapbench/
    env.py           – grid constants, random generation, transition model
    oracle.py        – future occupancy, value map, action delta, error metrics
    interventions.py – five intervention generators
    generate.py      – dataset assembly CLI
    validate.py      – assertion-based validation + diagnostics.json
    visualize.py     – matplotlib sample figures
    data.py          – PyTorch Dataset wrapper
models/
    erpm.py          – ErrorGatedPredictiveMapCNN, StandardPredictiveMapCNN
    baselines.py     – heuristic baselines
scripts/
    generate_data.py – config-driven data generation
    train.py         – training loop
    evaluate.py      – evaluation + baseline comparison
    make_plots.py    – paper-style figures
configs/
    smoke.yaml       – smoke test config (200 train, 2 epochs)
    pilot.yaml       – full pilot config (12k train, 30 epochs)
```

---

## Grid Representation

Each grid is `uint8 [C=10, H, W]`.

| Ch | Name |
|----|------|
| 0 | wall |
| 1 | start |
| 2 | goal |
| 3 | nuisance_visual |
| 4 | distractor |
| 5 | door_or_path_marker |
| 6 | one_way_up |
| 7 | one_way_down |
| 8 | one_way_left |
| 9 | one_way_right |

Actions: 0=up (−row), 1=down (+row), 2=left (−col), 3=right (+col).

---

## Label Semantics

`target_multihot: uint8 [4]` → `[sensory_update, value_remap, map_remap, action_remap]`

| Intervention | target |
|---|---|
| `sensory_nuisance` | `[1,0,0,0]` |
| `goal_relocation` | `[0,1,0,0]` |
| `topology_change` | `[0,0,1,0]` |
| `action_change` | `[0,0,0,1]` |
| `sensory_nuisance + action_change` | `[1,0,0,1]` |
| `sensory_nuisance + topology_change` | `[1,0,1,0]` |
| `goal_relocation + topology_change` | `[0,1,1,0]` |
| `goal_relocation + action_change` | `[0,1,0,1]` |

**`sensory_update` (index 0)**: the sensory/perceptual pathway should update, but the
predictive map (future-state structure) does not require remapping. This replaced
the earlier `no_remap` label, which was semantically misleading in composed settings.

---

## Oracle Definitions

### Transition model
`T[a, s]` = next state from state `s` under action `a`.
- Walls and out-of-bounds: agent stays.
- One-way cells: all actions forced to the one-way direction (first active channel wins).

### Future occupancy (successor representation)
Under a **uniform attempted-action** policy (each of the four actions is attempted
with probability 1/4; invalid moves result in staying — this matches the
transition model conventions). Solved via:

```
(I − γ·P)ᵀ x = e_{start}    →    x = M[start, :]
```

using `numpy.linalg.solve` (not explicit inverse). Normalised by max; γ=0.95.

> **Note**: the policy averages over *attempted* actions, not over *valid* actions.
> Invalid moves become self-loops in P. This is documented here and in `metadata.json`.

### Value map
Reverse BFS from goal under directed transitions.
`value[s] = γ^dist(s→goal)`, unreachable = 0.

### Action delta
`action_delta[a, r, c] = 1` iff `T_before[a, s] ≠ T_after[a, s]` for `s = r·W + c`.

### Error metrics

| Field | Computation |
|-------|-------------|
| `nuisance_error` | Mean L1 diff of channels 3+4 (nuisance_visual, distractor) |
| `full_visual_error` | Mean L1 diff of all 10 grid channels |
| `sensory_error` | Alias of `nuisance_error` (backward compat) |
| `future_error` | Mean `|delta_future|` |
| `value_error` | Mean `|delta_value|` |
| `action_error` | Fraction of free (action, cell) pairs where transition changed |

`action_error` uses free (non-wall) cells only to avoid dilution by wall fraction.

---

## Sample Fields

| Field | dtype | shape |
|-------|-------|-------|
| `before_grid` / `after_grid` | uint8 | [C,H,W] |
| `start_xy`, `goal_before/after_xy` | int64 | [2] |
| `intervention_id` | int64 | scalar |
| `target_multihot` | uint8 | [4] |
| `future_before/after`, `delta_future` | float32 | [H,W] |
| `value_before/after`, `delta_value` | float32 | [H,W] |
| `action_delta` | float32 | [4,H,W] |
| `nuisance_error`, `full_visual_error`, `sensory_error` | float32 | scalar |
| `future_error`, `value_error`, `action_error` | float32 | scalar |
| `path_len_before/after` | int64 | scalar |
| `layout_id`, `sample_id`, `layout_seed` | int64 | scalar |
| `intervention_seed`, `intervention_pair_id` | int64 | scalar |
| `weak_action_change` | int64 | scalar (0/1) |

---

## Dataset Splits

| Split | Default size | Intervention types | Seed offset |
|-------|--------------|--------------------|-------------|
| `train_single` | 12 000 | balanced single | 0 |
| `val_single` | 2 000 | balanced single | 500 000 |
| `test_single` | 2 000 | balanced single | 1 000 000 |
| `test_composed` | 2 000 | balanced composed | 1 500 000 |

Each sample uses a unique layout seed derived from its global index.
Disjoint seed offset ranges guarantee **no layout leakage across splits**.
`validate.py` verifies `layout_id` disjointness explicitly.

---

## Validation

```bash
python -m remapbench.validate --data_dir <data_dir>
```

Checks:
- NaN/Inf in oracle arrays
- Layout ID disjointness across splits
- Per-sample pass rates (≥ 90% required per class):
  - `sensory_nuisance`: high `nuisance_error`, near-zero future/value/action
  - `goal_relocation`: high `value_error`, near-zero future/action
  - `topology_change`: high `future_error`
  - `action_change`: high `action_error`
  - `composed`: ≥ 2 active target labels
- `weak_action_change` rate (warn if > 20%)
- `target_multihot` shape and label sums

Writes `diagnostics.json`.

---

## Training

```bash
python scripts/train.py --config configs/smoke.yaml --model erpm --seed 0
```

**Loss**:
```
future_weight · MSE(δfuture)
+ value_weight · MSE(δvalue)
+ action_weight · MSE(action_delta)
+ type_weight · BCEWithLogitsLoss(remap_logits, target_multihot)
```

Saves `results/{run_name}/best.pt`, `last.pt`, `train_log.jsonl`, `metrics_val.json`.

---

## Evaluation

```bash
python scripts/evaluate.py --config configs/smoke.yaml \
    --checkpoint results/smoke_erpm/best.pt --split test_single
```

Reports:
- Exact match, micro/macro F1, per-label precision/recall/F1
- False structural remap rate (sensory-only samples predicted with structural labels)
- Missed remap rates per structural label
- Δfuture / Δvalue MSE
- Per-intervention breakdown

Also evaluates **heuristic baselines**:
- `sensory_gated`: nuisance_error > threshold → sensory_update
- `value_gated`: value_error > threshold → value_remap
- `map_gated`: future_error > threshold → map_remap
- `action_gated`: action_error > threshold → action_remap
- `global_plasticity`: any error > threshold → activate all corresponding labels
- `oracle`: ground-truth target (upper bound)

---

## Plotting

```bash
python scripts/make_plots.py --config configs/smoke.yaml --run_dir results/smoke_erpm
```

Generates in `results/{run_name}/figures/`:
1. `error_dissociation_scatter.png` — nuisance vs future error, coloured by type
2. `baseline_false_structural_remap.png` — false structural remap rates
3. `missed_remap_rates.png` — missed remap by model/baseline
4. `remap_confusion_or_multilabel_heatmap.png` — per-label F1 heatmap
5. `sample_predictions.png` — before/after + true/predicted maps + labels

---

## Known Limitations

1. **Start-conditioned occupancy**: the future occupancy map is computed from the
   fixed start cell, not the full all-state predictive map. Future versions should
   store the full successor-representation matrix.

2. **Symbolic gridworld**: the environment uses multi-channel binary grids, not
   pixel observations. Extending to pixel rendering would stress-test sensory
   generalisation more realistically.

3. **Architecture-equivalent baselines**: `StandardPredictiveMapCNN` currently
   uses the same architecture as ERPM. Future work should strengthen the baseline
   (e.g., using a single scalar error head, or training without the type loss) to
   better ablate the error-gating mechanism.

4. **Shallow training**: 2-epoch smoke training does not converge; it is a
   pipeline check only. Run `configs/pilot.yaml` (30 epochs) for meaningful results.

5. **`action_change` behavioural relevance**: a `weak_action_change=1` flag marks
   fallback samples where path-relevance could not be confirmed. Monitor the weak
   rate (should be < 20%) in diagnostics.json.
