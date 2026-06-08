# RemapBench: Selective Remapping in Predictive Maps

This repository contains the RemapBench research codebase used to study a
specific failure mode in learned world models:

> When several things change at once, can a model identify which causal
> pathway changed, or does it collapse the update into a generic "something is
> different" signal?

The benchmark is built around small symbolic grid worlds. Each example contains
a `before` state, an `after` state, oracle predictive targets, and a four-bit
label saying which update pathways are active:

- `sensory_update`: appearance or nuisance features changed.
- `value_remap`: the goal or reward landscape changed.
- `map_remap`: topology or future-state structure changed.
- `action_remap`: affordances changed, such as a one-way transition.

The initial project implemented Error-Gated Plasticity for Selective Remapping
in Predictive Neural Maps. Later experiments in this repo focus on a harder
composition question: models can learn individual pathway changes, but often
fail when asked to generalize to a held-out pair of simultaneous changes.

## What This Repo Is For

Use this repo if you want to:

- generate audited RemapBench grid-world datasets;
- train direct classifiers and gated predictive-map models;
- test whether models generalize from seen single causes to held-out composed
  cause pairs;
- run diagnostic interventions such as oracle tuple decomposition, causal
  patching, evidence masks, C3 subset search, and slot decomposition;
- reproduce the saved final decomposition-gap result tables.

This is research code, not a polished library. The most important artifacts are
the dataset/audit scripts, model implementations, configs, and saved aggregate
tables under `results/`.

## Main Result Summary

The most recent final-paper aggregate is saved in
`results/final_paper_summary.json` and `results/final_paper_tables.txt`.

High-level interpretation:

- Direct models fit seen single/composed patterns but generalize poorly to
  held-out composed pairs. In the final aggregate, mean held-out exact match was
  low across direct models: about `0.197` for the standard CNN, `0.102` for
  gated ERPM, and `0.087` for independent gate heads.
- A privileged oracle-tuple diagnostic nearly closes the gap. This is not a
  fair baseline, because it uses decomposition information unavailable to the
  ordinary model, but it shows that the required information exists in the
  component transitions.
- Causal patching often rescues failures when the missing pathway is supplied,
  supporting the diagnosis that many failures are pathway-selection or
  composition failures rather than total representational collapse.
- Evidence-mask supervision and C3-style subset search are promising follow-up
  directions. Slot decomposition, as implemented here, did not solve the final
  composition problem.

Do not overread the final tables as a general claim about all world models. The
benchmark is symbolic, the interventions are controlled, and several diagnostics
are intentionally privileged to separate possible explanations.

## Repository Map

```text
remapbench/
  env.py                 grid constants and transition mechanics
  interventions.py       single and composed intervention generators
  oracle.py              future occupancy, value maps, action deltas
  generate.py            dataset assembly
  validate.py            structural validation
  planning.py            greedy value-following planning metrics
  data.py                PyTorch dataset wrapper

models/
  gated_erpm.py          explicit error-gated pathway model
  erpm.py                standard predictive-map CNN variants
  baselines.py           heuristic baselines
  counterfactual_difference.py
  slot_decomposition.py
  c3.py
  cpo.py
  fepo.py

scripts/
  generate_data.py       config-driven data generation
  audit_dataset.py       scientific validity audit
  train.py               original direct/gated training loop
  evaluate.py            original evaluation loop
  *_counterfactual_*     CDT / evidence-mask experiments
  *_evidence_mask*       evidence-mask audits, training, aggregation
  *_c3*                  C3 subset-search experiments
  *_slot_decomposition*  slot decomposition experiments
  final_*                final decomposition-gap run utilities
  modal_*                Modal launchers for larger runs

configs/
  smoke.yaml             small local smoke config
  pilot.yaml             original pilot config
  pilot_composed*.yaml   composed-training configs
  final_decomp_gap/      final-paper config matrix

docs/
  counterfactual_difference_pilot.md
  evidence_mask_pilot.md

results/
  final_paper_summary.json
  final_paper_tables.txt
  *_summary.json / *_tables.txt
```

## Core Dataset Idea

Each sample stores a pre-change and post-change grid plus oracle arrays:

- `future_before`, `future_after`, and `delta_future`;
- `value_before`, `value_after`, and `delta_value`;
- `action_delta`;
- scalar error diagnostics such as `future_error`, `value_error`, and
  `action_error`;
- a four-bit `target_multihot` label.

The four target bits are ordered as:

```text
[sensory_update, value_remap, map_remap, action_remap]
```

Common intervention labels:

| Intervention | Target |
|---|---:|
| `sensory_nuisance` | `[1,0,0,0]` |
| `goal_relocation` | `[0,1,0,0]` |
| `topology_change` | `[0,0,1,0]` |
| `action_change` | `[0,0,0,1]` |
| `goal_relocation + topology_change` | `[0,1,1,0]` |
| `sensory_nuisance + action_change` | `[1,0,0,1]` |
| `goal_relocation + action_change` | `[0,1,0,1]` |
| `sensory_nuisance + topology_change` | `[1,0,1,0]` |

Dataset generation uses disjoint layout IDs across splits. Always run the audit
before treating a generated dataset as scientifically usable.

## Quick Start

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run a local smoke pipeline:

```bash
python3 scripts/generate_data.py --config configs/smoke.yaml
python3 -m remapbench.validate --data_dir data/smoke
python3 scripts/audit_dataset.py --data_dir data/smoke --out data/smoke/audit_report.json --strict
python3 scripts/train.py --config configs/smoke.yaml
python3 scripts/evaluate.py --config configs/smoke.yaml \
  --checkpoint results/smoke_gated_erpm/best.pt \
  --split test_single \
  --gate_ablations
```

The smoke run is only a pipeline check. It is not evidence that the model has
learned the task.

## Original Pilot Workflow

Generate and audit the original pilot dataset:

```bash
python3 scripts/generate_data.py --config configs/pilot.yaml
python3 -m remapbench.validate --data_dir data/remapbench_v0
python3 scripts/audit_dataset.py \
  --data_dir data/remapbench_v0 \
  --out data/remapbench_v0/audit_report.json \
  --strict
```

Only train if the audit reports `PILOT READY: YES`:

```bash
python3 scripts/train.py --config configs/pilot.yaml
python3 scripts/evaluate.py --config configs/pilot.yaml \
  --checkpoint results/pilot_gated_erpm_seed0/best.pt \
  --split test_single \
  --gate_ablations
python3 scripts/evaluate.py --config configs/pilot.yaml \
  --checkpoint results/pilot_gated_erpm_seed0/best.pt \
  --split test_composed
python3 scripts/make_plots.py \
  --config configs/pilot.yaml \
  --run_dir results/pilot_gated_erpm_seed0
```

## Final Decomposition-Gap Workflow

The final-paper workflow lives mainly under `configs/final_decomp_gap/` and the
`scripts/final_*` utilities.

Useful commands:

```bash
python3 scripts/print_final_run_plan.py
python3 scripts/prepare_final_folds.py
python3 scripts/final_dataset_evidence_audit.py --strict
python3 scripts/modal_run_paper_experiments.py
python3 scripts/aggregate_paper_experiments.py
```

The saved aggregate expected all planned runs to complete before interpretation.
In the current saved final aggregate:

- `status` is `complete`;
- `missing_expected_reports` is empty;
- direct held-out performance remains low;
- oracle tuple and patching diagnostics show a large decomposition gap.

## Evidence-Mask and CDT Experiments

The counterfactual-difference and evidence-mask pilots ask whether explicit
primitive evidence supervision helps the model identify which pathway changed.

Start with:

```bash
python3 scripts/make_evidence_mask_configs.py
python3 scripts/audit_evidence_mask.py \
  --config configs/evidence_mask/fold1_sensory_action_bce_mask_seed0.yaml
python3 scripts/train_evidence_mask.py \
  --config configs/evidence_mask/fold1_sensory_action_bce_mask_seed0.yaml
python3 scripts/evaluate_evidence_mask.py \
  --config configs/evidence_mask/fold1_sensory_action_bce_mask_seed0.yaml \
  --checkpoint results/evidence_mask_fold1_sensory_action_bce_mask_seed0/best_val_composed_seen_exact.pt
python3 scripts/aggregate_evidence_mask.py
```

See `docs/evidence_mask_pilot.md` and
`docs/counterfactual_difference_pilot.md` for interpretation notes. The short
version: primitive masks are useful, but not uniformly sufficient; context and
readout choices matter.

## C3 and Slot-Decomposition Diagnostics

C3 treats the held-out composed prediction as subset selection over candidate
cause combinations. The saved C3 pilot summary reports that the best fair
delta-space variant closed the decomposition gap on held-out pairs, while
privileged factor-scored variants are diagnostics rather than fair baselines.

```bash
python3 scripts/make_c3_configs.py
python3 scripts/train_c3.py --config <config>
python3 scripts/evaluate_c3.py --config <config> --checkpoint <checkpoint>
python3 scripts/aggregate_c3.py
```

Slot decomposition tried to learn reusable pathway slots. The saved aggregate
decision is `SLOT_DECOMPOSITION_FAILED`; keep it as negative evidence and a
diagnostic scaffold rather than the main path.

```bash
python3 scripts/make_slot_decomposition_configs.py
python3 scripts/train_slot_decomposition.py --config <config>
python3 scripts/evaluate_slot_decomposition.py --config <config> --checkpoint <checkpoint>
python3 scripts/aggregate_slot_decomposition_pilot.py
```

## Validation and Audit Boundary

`remapbench.validate` checks structural correctness: array shapes, NaNs,
split layout IDs, target label shapes, and broad per-sample sanity thresholds.

`scripts/audit_dataset.py` is the stricter scientific gate. It checks split
completion, class balance, leakage, scalar signal profiles, topology/action
meaningfulness, composed evidence, metadata consistency, and channel-artifact
leakage.

For any serious run:

```bash
python3 -m remapbench.validate --data_dir <data_dir>
python3 scripts/audit_dataset.py --data_dir <data_dir> --out <audit.json> --strict
```

Do not train or report a generated dataset unless the strict audit passes.

## Known Limitations

- The environment is a symbolic multi-channel grid, not pixel observations.
- Some diagnostics are privileged by design and should not be described as
  deployable baselines.
- The greedy planning metric depends on learned value predictions and can be
  misleading for undertrained models.
- Saved local results include smoke/debug artifacts; prefer the named aggregate
  JSON/table files when summarizing conclusions.
- Several research branches live side by side in this repo, so read config
  paths and result filenames carefully before comparing numbers.
