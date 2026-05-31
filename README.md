# Error-Gated Plasticity for Selective Remapping in Predictive Neural Maps

**RemapBench v0 — pilot codebase**

---

## Project Summary

Biological brains distinguish between changes that require updating an internal
predictive map (topology changes, reward shifts) and changes that are merely
perceptual noise (lighting, texture). This pilot implements **Error-Gated Plasticity
(ERPM)**: a mechanism that routes different error signals — sensory, value, future-state,
and action prediction errors — to distinct update pathways, enabling *selective* remapping.

### GatedERPM Mechanism

`GatedERPM` implements explicit latent pathway routing via learned error gates:

1. **Encode** before/after grids into shared spatial latent states `z_before`, `z_after`.
2. **Error features** from `[z_b, z_a, z_a−z_b, |z_a−z_b|]` via a convolutional net.
3. **Gates** from global-average-pooled error features → MLP → `sigmoid` → `[B,4]`.
   The gates are simultaneously the `remap_logits` (trained via BCE against `target_multihot`).
4. **Pathway heads**: four independent convolutional heads compute pathway-specific
   latent updates from the error features.
5. **Gated update**: `z_update = Σ_k gate_k · u_k`, then `z_updated = z_before + z_update`.
6. **Decoders**: `future_after`, `value_after`, `delta_future`, `delta_value`, `action_delta`
   decoded from `z_updated`.

The coupling between gating and label prediction enforces mechanistic alignment:
if gate_k fires, the k-th pathway is active, and the BCE loss forces gate_k ≈ target_k.

| Pathway | Updates when | Oracle label |
|---------|-------------|--------------|
| Sensory | Appearance/nuisance changes (no structural shift) | `sensory_update` |
| Value | Goal/reward location changes | `value_remap` |
| Map | Topology / future-state structure changes | `map_remap` |
| Action | One-way / affordance constraints change | `action_remap` |

### Gate Ablation

Evaluate with `--gate_ablations` to run 7 modes: `zero_sensory`, `zero_value`,
`zero_map`, `zero_action`, `all_zero`, `all_one`, `oracle`. Results saved to
`gate_ablation_eval_{split}.json`.

Gate overrides affect both the latent pathway updates (via gated z_update) and the
reported remap predictions: overridden gates are converted back to logits via
`_safe_logit` so `remap_logits` is consistent with the ablated behavior. Oracle mode
should achieve perfect classification. Gate ablation evaluation reports classification
metrics, delta_future/value/action MSE, and planning behavior under each override.

### Planning Metric

A greedy value-following policy uses `value_after` predictions as a navigation signal.
At each step the agent moves to the highest-predicted-value non-visited neighbor.
Planning metrics: `planning_success_rate`, `mean_step_regret` vs oracle directed path,
`failure_rate`. Evaluation also records `planning_diagnostics` for predicted
`value_after`, oracle `value_after`, and stale `value_before`. If oracle-value
greedy planning is poor, de-emphasize the greedy planner metric itself.

---

## Quick Start

```bash
pip install -r requirements.txt
```

### Smoke / preflight commands (fast, run these first)

```bash
python scripts/generate_data.py  --config configs/smoke.yaml
python -m remapbench.validate    --data_dir data/smoke
python scripts/audit_dataset.py  --data_dir data/smoke --out data/smoke/audit_report.json
python -m remapbench.visualize   --data_dir data/smoke --out_dir figures/smoke --n 4
python scripts/train.py          --config configs/smoke.yaml
python scripts/evaluate.py       --config configs/smoke.yaml \
    --checkpoint results/smoke_gated_erpm/best.pt --split test_single --gate_ablations
python scripts/evaluate.py       --config configs/smoke.yaml \
    --checkpoint results/smoke_gated_erpm/best.pt --split test_composed
python scripts/check_gates.py    --config configs/smoke.yaml \
    --checkpoint results/smoke_gated_erpm/best.pt --split test_single
python scripts/make_plots.py     --config configs/smoke.yaml --run_dir results/smoke_gated_erpm
```

`validate.py` checks **structural correctness** — array well-formedness, broad
per-sample thresholds, NaN/Inf, layout-ID disjointness, and now prints
requested-vs-actual split sizes and composed pair counts.

`audit_dataset.py` checks **scientific representativeness** and is the strict
gate: a full pilot training run **must not start unless it reports
`PILOT READY: YES`** (run with `--strict` to make a failing audit exit nonzero).
It checks:
- **split completion** (requested vs actual saved counts; hard-fails if a split
  saved < 95% of requested samples — catches silent over-rejection in generation),
- **layout/sample leakage** (provably-disjoint layout IDs, unique sample IDs),
- **class balance** across interventions,
- **scalar signal profiles** per class (nearest-centroid separability sanity check),
- **topology/action meaningfulness** (behavioral effect, not cosmetic edits),
- **composed evidence** (each active component of a composed sample must be
  supported by its scalar errors, not just present in `target_multihot`),
- **action-metadata consistency** (one-way cell located for action samples,
  cleared for non-action samples),
- **channel / artifact leakage** (forbidden channels must not change per type).

### Pilot commands *(longer run; do not run by default)*

Before any full pilot run, generate, validate, and pass the strict audit:

```bash
python scripts/generate_data.py  --config configs/pilot.yaml
python -m remapbench.validate    --data_dir data/remapbench_v0
python scripts/audit_dataset.py  --data_dir data/remapbench_v0 \
    --out data/remapbench_v0/audit_report.json --strict
# Only proceed to training if the audit prints PILOT READY: YES
python scripts/train.py          --config configs/pilot.yaml
python scripts/evaluate.py       --config configs/pilot.yaml \
    --checkpoint results/pilot_gated_erpm_seed0/best.pt --split test_single --gate_ablations
python scripts/evaluate.py       --config configs/pilot.yaml \
    --checkpoint results/pilot_gated_erpm_seed0/best.pt --split test_composed
python scripts/make_plots.py     --config configs/pilot.yaml --run_dir results/pilot_gated_erpm_seed0
```

---

## Package Layout

```
remapbench/
    env.py           – grid constants, random generation, transition model
    oracle.py        – future occupancy, value map, action delta, error metrics
    interventions.py – five intervention generators
    generate.py      – dataset assembly CLI (supports test_larger, test_noisy)
    validate.py      – assertion-based validation + diagnostics.json
    visualize.py     – matplotlib sample figures
    data.py          – PyTorch Dataset wrapper
    planning.py      – greedy value-following policy + planning regret metrics
models/
    erpm.py          – ErrorGatedPredictiveMapCNN, StandardPredictiveMapCNN
    gated_erpm.py    – GatedERPM: explicit latent gating mechanism
    baselines.py     – heuristic baselines
scripts/
    generate_data.py – config-driven data generation
    audit_dataset.py – strict scientific-validity audit + pilot-readiness decision
    train.py         – training loop (supports gated losses)
    evaluate.py      – evaluation + baseline comparison + gate ablations + planning
    check_gates.py   – gate-override sanity check
    make_plots.py    – paper-style figures (gate heatmap, ablation, planning regret)
configs/
    smoke.yaml       – smoke test config (200 train, 2 epochs, gated_erpm)
    pilot.yaml       – full pilot config (12k train, 30 epochs, gated_erpm)
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
| `action_changed_count` | Count of changed free action transitions |
| `action_changed_cell_count` | Count of free cells with at least one changed action transition |
| `action_error_local` | Changed transitions divided by possible actions at changed cells |

`action_error` uses free (non-wall) cells only to avoid dilution by wall fraction,
but it is still grid-size dependent because the denominator grows with all free
cells. Scale-invariant action validity is based on `action_changed_count`,
`action_changed_cell_count`, and `action_error_local`.

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
| `action_changed_count`, `action_changed_cell_count` | int64 | scalar |
| `action_error_local` | float32 | scalar |
| `path_len_before/after` | int64 | scalar |
| `layout_id`, `sample_id`, `layout_seed` | int64 | scalar |
| `intervention_seed`, `intervention_pair_id` | int64 | scalar |
| `weak_action_change` | int64 | scalar (0/1) |
| `action_cell_row`, `action_cell_col` | int64 | scalar (−1 if not action_change) |
| `action_cell_on_path`, `action_cell_near_path` | uint8 | scalar (0/1) |
| `action_path_action_changed`, `action_path_len_changed` | uint8 | scalar (0/1) |

**Action-change relevance metadata** records *where* the one-way cell was placed
relative to the original shortest path (on-path / near-path) and *how* it changed
behavior (path-action redirected / directed path length changed). Non-action samples
use defaults (−1 / 0). The audit uses these to confirm action changes are
behaviorally meaningful rather than cosmetic local perturbations.

---

## Dataset Splits

| Split | split_id | Default size | Intervention types |
|-------|----------|--------------|--------------------|
| `train_single` | 0 | 12 000 | balanced single |
| `val_single` | 1 | 2 000 | balanced single |
| `test_single` | 2 | 2 000 | balanced single |
| `test_composed` | 3 | 2 000 | balanced composed |
| `test_larger` | 4 | 0 (1 000 in pilot) | balanced single, 12×12 |
| `test_noisy` | 5 | 0 (1 000 in pilot) | balanced single, noisy |
| `train_composed_seen` | 6 | 4 000 in composed pilot | composed pairs 0, 1, 3 |
| `val_composed_seen` | 7 | 1 000 in composed pilot | composed pairs 0, 1, 3 |
| `test_composed_seen` | 8 | 1 000 in composed pilot | composed pairs 0, 1, 3 |
| `test_composed_heldout` | 9 | 1 000 in composed pilot | held-out composed pair 2 |

`test_larger`: 12×12 grid (vs default 10×10), tests generalisation to unseen grid size.
`test_noisy`: higher nuisance density (`nuisance_prob=0.35`, `distractor_prob=0.25`).

The composed-training setup keeps global composed pair IDs stable:
`0=goal+topology`, `1=sensory+action`, `2=goal+action`,
`3=sensory+topology`. With `heldout_composed_pair_id: 2`, training and
validation see pairs `0,1,3`; `test_composed_heldout` measures generalization
to the unseen goal-plus-action composition. Legacy `test_composed` remains a
balanced aggregate over all pairs.

**Provably-disjoint layout IDs and seeds.** Layout IDs and RNG seeds are derived
arithmetically from `(seed, split_id, sample_index, attempt)`:

```
layout_id         = split_id * 1_000_000_000 + sample_index      # attempt-independent
base_seed         = seed * 10_000_000_000
layout_seed       = base_seed + split_id * 100_000_000 + i * max_sample_tries + attempt
intervention_seed = layout_seed + 50_000_000
```

Because `layout_id` depends only on `split_id` and `sample_index` (and indices are
far below the 1e9 stride), layout IDs occupy **non-overlapping billion-spaced ranges
per split** — collisions are impossible by construction, not merely checked after the
fact. The old additive `SEED_OFFSETS` scheme (which could theoretically collide) is
retained in metadata only as `seed_offsets_DEPRECATED`. `validate.py` and
`audit_dataset.py` still verify `layout_id` disjointness empirically.

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
  - `action_change`: changed action transition(s) with high local action error
  - `composed`: ≥ 2 active target labels
- `weak_action_change` rate (warn if > 10%, hard fail if > 25%)
- scale-invariant action stats when present: `action_changed_count`,
  `action_changed_cell_count`, and `action_error_local`
- `target_multihot` shape and label sums
- Optional robustness splits (`test_larger.npz`, `test_noisy.npz`) if present

Writes `diagnostics.json`.

---

## Training

```bash
python scripts/train.py --config configs/smoke.yaml
```

`--model` defaults to `cfg["model"]` (i.e., `gated_erpm` in the provided configs).
When `train_splits` and `val_splits` are configured, training concatenates the
listed `.npz` datasets without physically merging files. If omitted, the
backward-compatible defaults are `train_single` and `val_single`.

**Loss (GatedERPM)**:
```
future_weight        · MSE(δfuture)
+ value_weight       · MSE(δvalue)
+ action_weight      · MSE(action_delta)
+ type_weight        · BCE(remap_logits, target_multihot)
+ future_after_weight· MSE(future_after, future_after_target)
+ value_after_weight · MSE(value_after,  value_after_target)
+ gate_sparsity_weight · mean(gates)
+ stability_weight   · mean((gate_map * u_map)²)   ← sensory-only samples only
```

The stability loss penalizes map-pathway update magnitude *only on sensory-only samples*
(target = [1,0,0,0]) to avoid suppressing genuine structural remapping.
Logged as `loss_stability_map_on_sensory` in `train_log.jsonl`.

`train_log.jsonl` logs each loss component per epoch.
Saves `results/{run_name}/best.pt`, `last.pt`, `train_log.jsonl`, `metrics_val.json`.

---

## Evaluation

```bash
python scripts/evaluate.py --config configs/smoke.yaml \
    --checkpoint results/smoke_gated_erpm/best.pt --split test_single --gate_ablations
```

Reports:
- Exact match, micro/macro F1, per-label precision/recall/F1
- False structural remap rate (sensory-only samples predicted with structural labels)
- Missed remap rates per structural label
- Δfuture / Δvalue / action_delta MSE
- Planning metrics (`planning_success_rate`, `mean_step_regret`, `failure_rate`)
- Planning diagnostics for predicted, oracle-value, and stale-value maps
- Mean gate activations per intervention type
- Per-intervention breakdown

`--gate_ablations` additionally runs 7 ablation modes and saves
`gate_ablation_eval_{split}.json`.

Also evaluates **heuristic baselines**:
- `sensory_gated`: nuisance_error > threshold → sensory_update
- `value_gated`: value_error > threshold → value_remap
- `map_gated`: future_error > threshold → map_remap
- `action_gated`: action_error > threshold → action_remap
- `global_plasticity`: any error > threshold → activate all corresponding labels
- `oracle`: ground-truth target (upper bound)

### Next composed-training commands *(documented only)*

```bash
python3 scripts/generate_data.py --config configs/pilot_composed.yaml
python3 -m remapbench.validate --data_dir data/remapbench_v1_composed
python3 scripts/audit_dataset.py --data_dir data/remapbench_v1_composed \
    --out data/remapbench_v1_composed/audit_report.json --strict
# Train only after the strict audit reports PILOT READY: YES.
python3 scripts/train.py --config configs/pilot_composed.yaml
python3 scripts/train.py --config configs/pilot_composed_standard.yaml
python3 scripts/train.py --config configs/pilot_composed_ungated.yaml
python3 scripts/train.py --config configs/pilot_composed_global.yaml
python3 scripts/evaluate.py --config configs/pilot_composed.yaml \
    --checkpoint results/pilot_composed_gated_erpm_seed0/best.pt \
    --split test_composed_heldout
```

---

## Plotting

```bash
python scripts/make_plots.py --config configs/smoke.yaml --run_dir results/smoke_gated_erpm
```

Generates in `results/{run_name}/figures/`:
1. `error_dissociation_full_visual.png` — full_visual_error vs future_error by type
2. `error_dissociation_nuisance.png` — nuisance_error vs future_error by type
3. `gate_activation_by_intervention.png` — mean gate values heatmap per intervention
4. `gate_ablation_failures.png` — failure rate per ablation mode
5. `planning_regret.png` — mean step regret vs oracle
6. `baseline_false_structural_remap.png` — false structural remap rates
7. `missed_remap_rates.png` — missed remap by model/baseline
8. `remap_confusion_or_multilabel_heatmap.png` — per-label F1 heatmap
9. `sample_gate_predictions.png` — before/after grid + gate predictions

---

## Known Limitations

1. **Start-conditioned occupancy**: the future occupancy map is computed from the
   fixed start cell, not the full all-state predictive map. Future versions should
   store the full successor-representation matrix.

2. **Symbolic gridworld**: the environment uses multi-channel binary grids, not
   pixel observations. Extending to pixel rendering would stress-test sensory
   generalisation more realistically.

3. **Architecture-equivalent baselines**: `StandardPredictiveMapCNN` and legacy `erpm`
   use the same architecture. `GatedERPM` introduces a genuine gating mechanism.
   Use `model: gated_erpm` for mechanistic experiments.

4. **Shallow training**: 2-epoch smoke training does not converge; it is a
   pipeline check only. Run `configs/pilot.yaml` (30 epochs) for meaningful results.

5. **`action_change` behavioural relevance**: a `weak_action_change=1` flag marks
   fallback samples where path-relevance could not be confirmed. Generation now
   rejects weak action samples that have no path/value/future effect, and the audit
   hard-fails if the weak rate exceeds 0.20 or the meaningful fraction drops below
   0.50. Topology samples are likewise rejected unless they produce a real
   future/path/value change (audit requires ≥0.70 meaningful). Action-relevance
   metadata (`action_cell_*`) is stored per sample.

6. **Dataset audit is preflight, not proof of learnability**: `audit_dataset.py`
   confirms the *labels are dissociable from scalar errors and behaviorally grounded*
   (nearest-centroid separability ~0.85–0.92 on clean data — high but not a perfect
   artifact). It does not guarantee the neural model will learn the mapping; that is
   what the pilot run measures.

6. **Planning metric requires converged model**: the greedy value-following policy
   relies on `value_after` predictions. With untrained or 2-epoch smoke models the
   metric will reflect random walking, not learned planning.
