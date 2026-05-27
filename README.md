# RemapBench v0

Controlled gridworld dataset for *Error-Gated Plasticity in Predictive Neural Maps*.

Each sample is a `(before, after)` pair produced by applying one of five intervention regimes to a randomly generated gridworld. Ground-truth oracle labels encode which internal pathway a plasticity mechanism should update.

---

## Quick start

```bash
# Install dependencies (standard library + numpy + matplotlib only)
pip install numpy matplotlib

# Full dataset (~1 min on a normal CPU)
python -m remapbench.generate \
    --out data/remapbench_v0 \
    --seed 0 --grid_size 10 \
    --n_train 12000 --n_val 2000 --n_test 2000 --n_composed 2000

# Validate
python -m remapbench.validate --data_dir data/remapbench_v0

# Visualize
python -m remapbench.visualize \
    --data_dir data/remapbench_v0 --out_dir figures/remapbench_samples --n 8
```

**Smoke test** (as in the paper protocol):
```bash
python -m remapbench.generate --out data/smoke --seed 0 --n_train 200 --n_val 50 --n_test 50 --n_composed 50
python -m remapbench.validate --data_dir data/smoke
python -m remapbench.visualize --data_dir data/smoke --out_dir figures/smoke --n 4
```

---

## Package layout

```
remapbench/
    __init__.py        – public API
    env.py             – grid constants, random generation, transition model
    oracle.py          – future occupancy, value map, action delta
    interventions.py   – five intervention generators
    generate.py        – CLI + dataset assembly
    validate.py        – CLI + assertion checks + diagnostics.json
    visualize.py       – CLI + matplotlib figures
```

---

## Grid representation

Each grid is a `uint8 [C=10, H, W]` tensor.

| Channel | Meaning |
|---------|---------|
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

## Intervention regimes

| ID | Name | target_multihot | Oracle signal |
|----|------|-----------------|---------------|
| 0 | `sensory_nuisance` | `[1,0,0,0]` | high `sensory_error`, near-zero future/value/action |
| 1 | `goal_relocation` | `[0,1,0,0]` | high `value_error`, near-zero future/action |
| 2 | `topology_change` | `[0,0,1,0]` | high `future_error` |
| 3 | `action_change` | `[0,0,0,1]` | high `action_error` |
| 4 | `composed` | multi-hot | combination of above |

`target_multihot` is `uint8 [4]` for `[no_remap, value_remap, map_remap, action_remap]`.

---

## Sample fields

| Field | dtype | shape | Description |
|-------|-------|-------|-------------|
| `before_grid` | uint8 | [C,H,W] | Grid before intervention |
| `after_grid` | uint8 | [C,H,W] | Grid after intervention |
| `start_xy` | int64 | [2] | Start position (x=col, y=row) |
| `goal_before_xy` | int64 | [2] | Goal position before |
| `goal_after_xy` | int64 | [2] | Goal position after |
| `intervention_id` | int64 | scalar | Regime index |
| `target_multihot` | uint8 | [4] | Multi-hot plasticity label |
| `future_before/after` | float32 | [H,W] | Successor representation from start |
| `delta_future` | float32 | [H,W] | `future_after − future_before` |
| `value_before/after` | float32 | [H,W] | Oracle value map (γ^dist to goal) |
| `delta_value` | float32 | [H,W] | `value_after − value_before` |
| `action_delta` | float32 | [4,H,W] | 1 where transition changed, else 0 |
| `sensory_error` | float32 | scalar | Mean L1 diff of nuisance/distractor channels |
| `future_error` | float32 | scalar | Mean L1 diff of future occupancy maps |
| `value_error` | float32 | scalar | Mean L1 diff of value maps |
| `action_error` | float32 | scalar | Fraction of free (action,cell) pairs that changed |
| `path_len_before/after` | int64 | scalar | Directed shortest path length (−1=unreachable) |

---

## Oracle details

### Future occupancy
Successor representation under uniform random valid actions. Built by constructing the
row-stochastic transition matrix `P` and solving `(I − γP)ᵀ x = e_start` via
`numpy.linalg.solve` (not explicit inverse). Normalized by max; wall cells set to 0.
Default γ = 0.95.

### Value map
Backward BFS from goal over directed transitions (respecting one-way tiles).
`value[s] = γ^dist(s, goal)`, unreachable states = 0.

### Action delta
Binary heatmap: `action_delta[a,r,c] = 1` iff the directed transition from cell (r,c)
under action `a` differs between before and after grids.

### Sensory error
Mean L1 difference over the nuisance_visual (ch 3) and distractor (ch 4) channels.

### Action error
Fraction of free-cell × action pairs where transition changed. Computed over non-wall
cells only to avoid dilution by the wall fraction.

---

## Split design

| Split | Size | Interventions | Seed offset |
|-------|------|--------------|-------------|
| `train_single` | 12 000 | balanced single | 0 |
| `val_single` | 2 000 | balanced single | 500 000 |
| `test_single` | 2 000 | balanced single | 1 000 000 |
| `test_composed` | 2 000 | balanced composed pairs | 1 500 000 |

**Leakage rule**: each sample uses a unique layout seed derived from its
position in the global index. Disjoint seed offset ranges guarantee that
no base layout is shared across splits.

**Composed pairs** (balanced in `test_composed`):

| Pair | `target_multihot` |
|------|-------------------|
| goal_relocation + topology_change | `[0,1,1,0]` |
| sensory_nuisance + action_change | `[1,0,0,1]` |
| goal_relocation + action_change | `[0,1,0,1]` |
| sensory_nuisance + topology_change | `[1,0,1,0]` |

---

## Output files

```
data/remapbench_v0/
    train_single.npz
    val_single.npz
    test_single.npz
    test_composed.npz
    metadata.json        – channel map, composed pairs, config
    diagnostics.json     – per-split stats and assertion results
```

---

## Design choices (ambiguities resolved)

- **One-way tiles**: all four actions from a one-way cell are redirected to the
  one-way direction (first active channel wins). This is the most conservative
  interpretation of "force/allow only up."
- **Future occupancy normalization**: normalized by max value (not sum) so the
  map is a relative visitation density on [0,1].
- **Action error denominator**: free cells only, to keep the signal scale
  independent of wall density.
- **Topology change "open wall"**: opened cells receive `CH_DOOR=1` as a marker.
- **Composed samples are test-only by default**, as specified.
- **Post-hoc oracle rejection**: samples that fail minimum oracle error thresholds
  are rejected and retried (up to `max_sample_tries=50`). This ensures every
  sample has a detectable signal for its label.
