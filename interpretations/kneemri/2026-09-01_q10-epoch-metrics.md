# Q10 (knee-5dfh): what we track per epoch, and which missing metrics would have changed a decision

2026-09-01, Kimi. Audit by a read-only agent, every claim file:line-checked. The headline first:
**the bead's premise is stale** — per-label AUC is already recorded per epoch, and has been since the
recorder's layer split (commit fa6e9a16). What is missing is not the field but a per-label
*trajectory reader*: `run_comparison` reads only the best row's `per_label`
(`kneemri/evaluation/run_comparison.py:219-220`).

## What is recorded per epoch today

`EpochRow` (`kneemri/runs/run_metrics.py:46-80`): `epoch`, `loss`, `auc`, `seconds`,
`per_label: dict[str, float]` (the twelve holdout AUCs, `kneemri/training/fitting.py:123-124` via
`LabelScores.per_label`), plus `NotRequired` `vram_peak_gb` / `vram_alloc_gb` / `started`. JSONL, one
object per line; the headline `auc` is the nanmean over however many labels were defined. Full
predictions + truth persist in `.holdout.npz` (`kneemri/runs/holdout_record.py:26-46`), so any
per-study metric is also computable offline at zero marginal cost.

## The four candidates, against past decisions

| metric | status | a decision it would have altered |
|---|---|---|
| per-label AUC | **already recorded per epoch** | the blendlabels read is the proof it matters: mean +0.0034 hid 7 helped / 5 diluted labels (MCL +0.0239, Lateral Meniscus −0.0105), ledger `experiments/README.md:325` (bead says 6/6; the ledger reads **7/5**). The missing piece is a per-epoch per-label READER — zero recording cost |
| AUPRC | computed nowhere (zero grep hits for average_precision/auprc/precision_recall) | **none on record.** The Fracture arms each "improved" it inside a ±0.061 band (ledger:1063,1071,4129), but at 62 holdout positives AUPRC is at least as noise-dominated as the AUC it replaces. Diagnostic value (top-of-ranking on rare labels), not decisional |
| calibration error (ECE) | no member-side instrument (`level_calibration.py` calibrates the report READER against 58 gold studies; `inference/ensemble.py:6` decided calibration is worth nothing because the metric is rank-based) | **the divergence/seat readings**: `member_divergence.py:30-32` names the defect (disagreeing only about calibration scores high on divergence), and ledger:4534/4450 record `--calibrated-targets` buying disagreement while moving quality zero. A per-epoch ECE would have said whether that disagreement was calibration or opinion. Caveat: truth is binarised at 0.5 for scoring (`scores.py:20`) |
| per-example loss distribution | not recorded — `fit_epoch` accumulates a scalar on-device (`trainer.py:440,517`); elementwise BCE exists (`objective.py:270`) but is reduced before return | **the disagreed-sampler premise**: `train_disagreement.py:13-19` rejected loss as the hard-example signal on the strength of `uncertainty_split`'s ~87% aleatoric estimate — an entropy-decomposition argument, never a measured loss distribution of our own runs. Recording quantiles per epoch tests that premise directly. Secondary: the loss-falls-while-AUC-flats overfit shape (ledger:303) is today readable only as a scalar |

## The touch list (for whoever lands the code — Claude's tree)

- Schema: `run_metrics.py:46-80` — new fields `NotRequired` (line 70's discipline: pre-existing files must still read).
- Writer: `fitting.py:216-225` — row construction; AUPRC/ECE belong in `ranked` (`fitting.py:111-124`) where predictions and truth are already in hand.
- Backfill constructor: `kneemri/evaluation/holdout_metrics.py:59-65` — NaN-fill any new required field (its test asserts exactly this discipline for `loss`).
- Per-example loss quantiles additionally touch `trainer.py:440-527` (`fit_epoch` return + heartbeat).
- Readers to surface new fields: `trajectory_check.py:159-165`, `convergence_check.py:194`,
  `ledger_check.py:104`, `claims_check.py:351-353`, `budget_residuals.py:67`, `progress.py:52`,
  `run_comparison.py:219-220`; mirror tests under `tests/unit/kneemri/{runs,training}/`.

## The answer in one line

Add **ECE** and **per-epoch loss quantiles** (each names a real past decision), skip **AUPRC** (no
decision at 62 positives), and for per-label AUC spend nothing on recording and a little on a
per-epoch per-label READER — the data has been sitting in the metrics files all along.
