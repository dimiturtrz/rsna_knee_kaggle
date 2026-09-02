# Training-pipeline audit — problems, simplifications, improvements

Date: 2026-09-02  
Scope: `kneemri/training/` and `kneemri/models/` as used by the supervised and pre-training recipes.
Method: read-only review of the code that builds and runs fits; no card spent.

## Context worth keeping in mind

- The project is on a 0.937–0.941 board plateau. The champion is a monoculture of DINOv2-based dense members, and most ensemble-diversity axes have closed inside 0.005.
- DINOv2 outperforms DINOv3/OrthoFoundation-L **with the current readout**, which is surprising. A frozen readout probe (`experiments/arms/readout_probe.sh`) is actively testing whether the gap is the trunk or the pooling.
- The external-data lane (MRNet, Rijeka, fastMRI) is still open; bead `knee-mwuv` proposes a unified mixed-pretraining scheme.
- The closed-question backlog Q-AA through Q-AO needs owner/Claude decisions; this audit does not address those directly but flags where training-code work depends on them.

---

## 1. High-impact, actionable problems

### 1.1 `MarginObjective` silently skips rare labels in most batches

**Where:** `kneemri/training/objective.py:582-607`

A label needs both a positive and a negative in the batch to form the AUC-margin term. With batch 12 and Fracture at ~7% prevalence, the term is skipped in roughly half the steps. The docstring acknowledges this and calls it a batch-size artefact, but the artefact is **a real bias**: the rarest labels get less gradient than common ones, exactly the opposite of what the margin mechanism is supposed to help.

`RememberedRanking` and `RememberedContrast` already solved the analogous problem for ranking/contrastive terms by keeping a `PairMemory` of past readouts. There is no `RememberedMargin`.

**Fix:** add a remembered margin objective that accumulates per-label positive/negative pools across the last N steps, then computes the class means and shortfall against the pooled population. Gradient still flows only through the live batch; the remembered side is fixed. This re-uses `PairMemory` / `kneemri/training/pair_memory.py`.

**Suggested bead:** implement `RememberedMargin` behind `--pair-memory` when `--auc-margin` is set, mirroring `RememberedContrast`.

### 1.2 Disabled augmentation ops still run and allocate

**Where:** `kneemri/training/augmentation.py:148-202`

When the default `geometric_only` config is used, `_intensity` still computes gamma (`pow`), blur (`conv2d`), and contrast transforms and then multiplies the results by zero. The docstring says this costs ~15.5 ms vs 2.0 ms and ~3.2 GB transient allocation per step on the whole-stack shape. The ops are also on the default path of every historic arm.

The stated reason for not short-circuiting is RNG reproducibility: a skipped op advances the generator differently. That is a real cost, but it is paid once per arm re-run, while the wasted compute is paid on every step of every run.

**Fix options:**
- (cheap, reproducibility change) add early `if` returns when a sub-op is disabled and accept that a rerun gets a different realisation of the same distribution. Document in the arm header.
- (cheap, no reproducibility change) fuse the disabled path into a single no-op that still advances the generator by the same number of draws.

Either is a clear simplification/perf win; the current “compute and discard” path exists only because the two were not disentangled.

### 1.3 Slot dropout creates a train/test distribution mismatch

**Where:** `kneemri/training/augmentation.py:204-234`

Training may randomly withhold present slots; inference always sees every acquired slot. The head masks absent slots, so the model can handle absence, but it was never trained on the *distribution* of absences that appears at inference (~55–60% fill for some slots). This is the supervision-axis regulariser the feature was added for, but the regulariser is only half-applied if inference never simulates the same missing-slot distribution.

**Fix:** add an inference-time slot expectation: run the model with multiple slot-dropout masks and average the logits, or at minimum measure whether deterministic “drop the weakest-attended slot” changes board predictions. If it helps, the train/test gap is real and cheap to close.

---

## 2. Design choices that are live, not broken

### 2.1 Spectral MAE pretraining draws slices independently

**Where:** `kneemri/training/pretrain.py:126-140`

Slices are sampled independently across studies and slots. This avoids the neighbour-copying problem the docstring describes, but it also throws away all study/slot structure. For a reconstruction objective that may be fine; for learning knee-specific anatomy it may be suboptimal.

**Suggestion:** run a cheap ablation where a fraction of each batch is drawn as whole slots (or even whole studies) while still masking patches. If the slot-structured batch learns faster or transfers better, make it an option.

### 2.2 Spectral MAE decoder is a single linear

**Where:** `kneemri/training/spectral_mae.py:111`

The decoder is intentionally linear because the encoder reads the full token sequence. The project has not measured whether a small transformer decoder helps reconstruction or transfer. Given that the pretraining lane is still open, this is a free hyper-parameter to test once a baseline spectral-MAE checkpoint is proven.

### 2.3 `through_plane` is hard-coupled to a 3-channel window

**Where:** `kneemri/models/model.py:87-108`, `kneemri/models/slice_read.py`

`through_plane` feeds (i-1, i, i+1) into the three RGB channels. That only works for a window of exactly three slices and only makes sense for 2D trunks. A true volumetric stem would take an arbitrary depth window with a `Conv3d` or a tubelet. The current coupling is the cheapest test of the idea, but it cannot be swept over window size.

**Suggestion:** if the readout probe shows the trunk is not the bottleneck, a small `Conv3d` patch-embedding stem (or a configurable depth convolution prefix) is the next volumetric experiment, not more channel-stacking arms.

---

## 3. Simplifications

### 3.1 Objective dispatch in `run_setup.py` is at complexity ceiling

**Where:** `kneemri/training/run_setup.py:193-279`

The `objective()` chain is an `if` ladder with six term classes and a separate `_check_composed()` method that duplicates the composition rules. The guards are excellent, but the dispatch is hard to extend and the rules live in two places.

**Suggestion:** introduce an `ObjectiveBuilder` that owns the composition graph (which terms can coexist, which replace cross-entropy, which need `pair_memory`, which need `widths`). Each term registers itself with its prerequisites; `objective()` walks the graph once. This would make adding a `RememberedMargin` a one-line registration instead of another branch.

### 3.2 `TrainingOptions` is a flat bag of ~90 fields

**Where:** `kneemri/training/options.py:50-183`

The dataclass is flat because argparse dest names land as scalar fields. The rest of the code already groups them via properties (`crop`, `band`, `reading`, `supervision`, `positions`). The LCOM4=6 advisory is correctly ignored in the comment, but the type is still awkward to construct in tests and hard to read.

**Suggestion:** keep the flat parser target as a private `_RawOptions`, then construct a nested `TrainingConfig` (or use `replace` + properties) for downstream use. This is a refactor, not a behaviour change, and it would make `RunSetup` much clearer. Risk: every arm and manifest uses `TrainingOptions.of_recipe()`; migration must preserve every field name.

### 3.3 `_readout_parameters()` identifies head params by exclusion

**Where:** `kneemri/models/model.py:498-513`

Every parameter not in `backbone.parameters()` is treated as head group. This fixed the earlier bug where `SliceAttention` was a sibling of the head and never trained. It is fragile in the opposite direction: any future module inserted between encoder and head (e.g. a learned readout projection) will silently get the head LR.

**Suggestion:** make the grouping explicit. `StudyModel` could expose `self.readout_modules` (head, slice aggregator, readout projection) and the trainer uses that instead of exclusion. Low urgency while the architecture is stable.

---

## 4. Things that are correct and should not be touched

- **EMA scoring:** `Fitting.ranked()` is called inside `Trainer.evaluated()`, and `keep()` clones the state that is saved. The epoch is ranked on the weights that are written.
- **Wrapper order:** `RunSetup.model()` applies LoRA → recompute → compile in the right order; compile after LoRA is required.
- **Whole-stack guard:** `StudyModel.check_whole_stack()` correctly distinguishes per-slice pooling from tubelet trunks and rejects channel stacking.
- **Half-guards for teachers/reports:** `RunSetup.teacher()`, `teacher_embeddings()`, and `reports()` raise before the card is touched when only one half of a mechanism is present.
- **Certainty ramp:** `_ramped_weights()` preserves total mass, so loss scale matches the un-ramped control.
- **No horizontal flips:** augmentation correctly avoids the laterality swap that would corrupt five of the twelve labels.

---

## 5. Recommended next actions

1. **Implement `RememberedMargin`** — highest-impact training-code change; directly addresses the rare-label deficit and re-uses existing machinery. Smoke against `calib_dense_hold` or a small dense arm.
2. **Short-circuit disabled augmentation ops** — immediate throughput/memory win; decide whether to preserve RNG by faking draws.
3. **Test inference-time slot expectation** — cheap CPU/light-GPU experiment; if it helps, it changes the inference path, not training.
4. **Run the readout probe** (`experiments/arms/readout_probe.sh`) before opening any new trunk lane; the “DINOv2 is best” result may be a readout artefact.
5. **Proceed with `knee-mwuv` mixed pre-training** as the external-data lane; the training-pipeline changes needed (masked BCE, mixed sampler, domain token) are smaller than new architecture bets.
6. **Refactor `ObjectiveBuilder`** only after the margin fix lands; doing it before adds churn to the same file.

---

## 6. Beads to file

- `knee-training-remembered-margin` — implement remembered AUC-margin objective for rare labels.
- `knee-training-augment-shortcircuit` — stop computing disabled intensity ops.
- `knee-training-slot-dropout-inference` — close the train/test missing-slot gap.
- `knee-training-objective-builder` — refactor `run_setup.objective()` dispatch into a builder/graph.
