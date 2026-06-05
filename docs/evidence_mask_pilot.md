# Evidence-Mask Supervision Pilot

This pilot refocuses the previous counterfactual-difference experiment around
primitive evidence-mask supervision.

The model remains compact: `DiffCauseNet` takes `before`, `after`,
`after-before`, and `abs(after-before)` channels and predicts four cause labels
plus four primitive evidence maps.

## Why CDT Is Deprioritized

The first fold showed the large gain came from primitive mask supervision:
BCE-only failed, while BCE plus primitive masks nearly matched the full
context-loss run. That made CDT too complex as the default story.

The second audited fold changed the nuance: primitive masks alone were not
enough for sensory+action, while the context ablation recovered high heldout
accuracy. CDT is therefore retained as an ablation, not the main method.

## Primitive Diff Rule

`primitive_diff_rule` is a rule baseline from primitive channel diffs, not an
oracle for target labels. It predicts a cause as active when the associated
channel group has any nonzero grid diff:

- sensory: channels `3,4`
- value: channel `2`
- topology/map: channels `0,5`
- action: channels `6,7,8,9`

Primitive diffs can disagree with generator labels because visible channel
changes and causal target labels are not identical.

## Commands

Generate evidence-mask configs:

```bash
python3 scripts/make_evidence_mask_configs.py
```

Audit, train, and evaluate one run:

```bash
python3 scripts/audit_evidence_mask.py --config configs/evidence_mask/fold1_sensory_action_bce_mask_seed0.yaml
python3 scripts/train_evidence_mask.py --config configs/evidence_mask/fold1_sensory_action_bce_mask_seed0.yaml
python3 scripts/evaluate_evidence_mask.py --config configs/evidence_mask/fold1_sensory_action_bce_mask_seed0.yaml --checkpoint results/evidence_mask_fold1_sensory_action_bce_mask_seed0/best_val_composed_seen_exact.pt
```

Aggregate:

```bash
python3 scripts/aggregate_evidence_mask.py
```

## Current Trusted Local Results

Trusted audited folds available locally: fold0 and fold1.

Untrusted or blocked:

- fold2 local data is corrupt/truncated.
- fold3 local path is not an audited fold directory.
- clean fold2/fold3 exist on Modal volume `iconip-monkey-final-decomp-gap-data`.

Current interpretation:

- Fold0: evidence masks close most of the gap.
- Fold1: evidence masks alone fail; context/mixed-union recovers performance.
- The main direction should be evidence-mask supervision with context retained
  as an explicit ablation until clean fold2/fold3 decide whether it is broadly
  necessary.
