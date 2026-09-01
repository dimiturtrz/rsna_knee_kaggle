# Q10 (knee-5dfh): what we track per epoch, and which missing metrics would change a decision

2026-09-01, Kimi. Read-only audit, every claim file:line-checked.

## Headline

The epoch record already carries more than the bead assumed. **ECE** and **per-study loss quantiles** (`loss_q25`..`loss_q95`) are written by `fitting.py` and tested in `tests/unit/kneemri/runs/run_metrics.py:354-376`. What is still missing are: a **per-label AUC trajectory reader** (per-label AUC is recorded every epoch but only the best epoch is compared), **holdout/validation loss** (the only overfit signal now is AUC declining while train loss falls), **per-label calibration**, **AUPRC for rare labels**, and **optimizer state** (LR, gradient norm).

## What is recorded per epoch today

`EpochRow` (`kneemri/runs/run_metrics.py:46-93`):

- required: `epoch`, `loss`, `auc`, `seconds`, `per_label: dict[str, float]`
- optional: `vram_peak_gb`, `vram_alloc_gb`, `started`, `ece`, `loss_q25..loss_q95`

Where each piece comes from:

- `per_label` — the twelve holdout AUCs, computed by `LabelScores.per_label` (`kneemri/evaluation/scores.py:54`) and returned from `Fitting.ranked` (`kneemri/training/fitting.py:144-157`).
- `ece` — 15 equal-width bins, truth binarised at 0.5, computed in `Fitting.epochs` (`kneemri/training/fitting.py:244-248`).
- `loss_q*` — per-study BCE quantiles from `Trainer._quantiles_of` (`kneemri/training/trainer.py:388-401`), collected during `fit_epoch` (`kneemri/training/trainer.py:577-583`) and written in `Fitting.epochs` (`kneemri/training/fitting.py:249-269`).
- VRAM — attached by `RunMetrics.record` from CUDA reserved/allocated (`kneemri/runs/run_metrics.py:226-229`).

## What the files on disk actually contain

Only three production metrics files exist under `logs/` (`band_full_partial_3ep`, `keep_band_full_run1_partial`, `half_data_plumbing`) plus two smoke files in the repo root. None of them contain `ece` or `loss_q*`. The schema's `NotRequired` discipline (`kneemri/runs/run_metrics.py:85-93`) is what lets old and new rows coexist. So the **instrumentation is landed but unexercised** by recent full runs; any new run will write the full set.

## Missing metrics that would change a decision

| missing metric | why it matters | decision it would have altered |
|---|---|---|
| **per-label AUC trajectory reader** | `per_label` is stored every epoch, but `run_comparison.py:216-223` only reads the best epoch's vector | `blendlabels` (README:325) helped MCL +0.0239 and diluted Lateral Meniscus −0.0105; a reader that showed when each label peaked or declined would separate real supervision gains from epoch noise |
| **holdout/validation loss** | only train `loss` is recorded; overfit arms show AUC falling while train loss still falls | `attend_true` (README:302), `depth_true` (README:303), and `aug_long` (README:328/1342) all have the converge-and-overfit shape; a held-out loss gap would quantify it and could drive early-stopping or regulariser decisions |
| **per-label ECE / calibration** | current ECE is pooled across all labels | `calibrated_targets` inverted on Effusion −0.0212 and PF OA +0.0170 (README:318); per-label calibration would show whether the remap helped the labels it targeted |
| **AUPRC for Fracture** | Fracture has only ~62 holdout positives, so AUC is noisy | past Fracture arms (README:1063,1071,4129) moved inside ±0.061; AUPRC would rank top-of-list performance, which is what matters for a rare finding |
| **gradient norm + learning rate** | schedule and clipping are policies, but their effect is not logged | would debug `one_cycle` stepping and `GradientClip` interactions; the `encoder_adapt_thawed` collapse (README:304) might have shown exploding gradients |
| **slot coverage / dropout count** | `slot_dropout` is a policy, but no per-epoch count of dropped slots or missing-slice studies is logged | would gate coverage arms (`extended_slots`, `crop80_span`) by verifying the intended anatomy actually arrived |

## Verdict

- Already done: ECE and per-study loss quantiles are in the record and tested.
- Highest leverage next: (1) a per-label AUC trajectory reader (zero recording cost), (2) holdout loss, (3) per-label ECE, (4) Fracture AUPRC.

## Touch points for whoever implements

- Reader: `kneemri/evaluation/run_comparison.py:216-223` plus a new trajectory report.
- Holdout loss: add to `Fitting.ranked`/`Fitting.epochs`; the objective already computes per-example BCE under `torch.no_grad()` in `trainer.py:577-583`, but that is train-only. Holdout loss needs a second forward on `fold.holdout`.
- Per-label ECE: refactor `_calibration_error` in `kneemri/training/fitting.py:120-142` to return per-label errors.
- AUPRC: add to `kneemri/evaluation/scores.py` beside `per_label`.
- Gradient/LR: log optimiser param-group LR and the return value of `torch.nn.utils.clip_grad_norm_` from `GradientClip`.
- Slot coverage: count masked slices per batch in `Trainer.fit_epoch` or derive from `corpus.mask`.
