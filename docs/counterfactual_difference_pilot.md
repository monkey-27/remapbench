# Counterfactual Difference Training Pilot

This pilot tests whether cause detectors generalize across context by training on
linked tuple transitions from existing RemapBench final-fold data.

The implemented method is intentionally compact:

- `counterfactual_difference` / `DiffCauseNet`
- input: `before`, `after`, signed diff, absolute diff
- outputs: four cause logits, four evidence maps, four evidence vectors
- loss: BCE labels, optional primitive masks, `context_difference_loss`,
  mixed-union consistency, inactive evidence suppression

The current local dataset contains `base/A/B/AB` tuples only. There is no
persisted `BA` role, so the context loss currently covers the valid
`E_B(base->B) ~= E_B(A->AB)` direction. Audit reports mark this as not fully
symmetric.

## Commands

Generate configs:

```bash
python3 scripts/make_cdt_configs.py
```

Smoke audit/train/eval:

```bash
python3 scripts/audit_counterfactual_difference.py --config configs/cdt/smoke.yaml
python3 scripts/train_counterfactual_difference.py --config configs/cdt/smoke.yaml --max-tuples 256 --epochs 2
python3 scripts/evaluate_counterfactual_difference.py --config configs/cdt/smoke.yaml --checkpoint results/cdt_smoke/best_val_composed_seen_exact.pt
```

One local fold:

```bash
python3 scripts/audit_counterfactual_difference.py --config configs/cdt/fold0_goal_topology_full_seed0.yaml
python3 scripts/train_counterfactual_difference.py --config configs/cdt/fold0_goal_topology_full_seed0.yaml
python3 scripts/evaluate_counterfactual_difference.py --config configs/cdt/fold0_goal_topology_full_seed0.yaml --checkpoint results/cdt_fold0_goal_topology_full_seed0/best_val_composed_seen_exact.pt
```

Aggregate downloaded/local reports:

```bash
python3 scripts/aggregate_counterfactual_difference.py
```

## Interpretation Rules

Treat high seen-pair accuracy as capacity only. The method matters only if
`test_composed_heldout` improves without oracle routing or tuple decomposition.

The primitive masks are benchmark-specific supervision, not a general method
claim. A convincing follow-up should add audited `BA` support and run all four
folds with the same config matrix.
