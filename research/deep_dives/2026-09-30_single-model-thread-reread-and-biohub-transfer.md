# Topic 735304 re-read (Sep posts) and what the biohub post-mortem transfers — 2026-09-30

Read 2026-09-30 17:13Z through Kaggle's internal `GetForumTopicById` endpoint (the page is JS-rendered).
It was last swept on 2026-08-26 ([rival_single_models](2026-08-26_rival_single_models_and_the_label_bottleneck.md)).
Every number below is a competitor's self-report on the PUBLIC board, not a measurement of ours.

## New self-reported single-model scores since 08-26

| who | claim | recipe as stated |
|---|---|---|
| CoolinLai (09-28) | 0.954 | five-fold ResNet at 224 px. Five folds, so this is not a solo in our sense |
| NguyenThanhNhan (09-21) | 0.950, single fold, ~3.5 h | **Qwen 3.5 2B** fine-tuned at 384×384, LoRA on the LM jointly with the vision encoder |
| Archit Konde (09-16) | 0.950, single fold | 2.5D CoAtNet at 224, a few neighbouring slices. OOF 0.930 on gold-58 |
| Raymond Yuen (09-15 → updates) | 0.938 → 0.940 → 0.949, full train | CoAtNet 288. "still label" at every step |
| Scott Willis (09-05) | 0.949, single fold | small ResNet, own Gemma-4 labels (~0.89 vs gold), ~0.930 on gold-58, ~5 min inference |
| tennogh (09-03) | 0.942 raw / 0.943 TTA | 288 px. "OOF pseudo-labels have been pretty well correlated with LB" |
| Yann Majewski (09-01) | 0.936, single fold | small ResNet 224. Label merge with a public table +0.015, then encoder pretraining + longer/regularised |

**Correction to [labels summary §8](../summaries/labels-and-label-noise.md):** that section says a quoted
"0.949 small ResNet" is not in the thread and was withdrawn. It IS in the thread now: Scott Willis,
2026-09-05, "My current LB best (0.949) is a single fold, single model".

## The one mechanism the thread converges on: out-of-fold soft bootstrapping of the TARGET

Archit Konde, 09-18 and 09-21, stated in full:

- Masking noisy cells did nothing. Replacing labels with predictions was worse than **mixing** them.
- Keep labels as fractions; rounding to 0/1 hurt.
- The gain came from the "silent" cells: reports that never mention a finding (Synovitis most of all).
- **The predictions must be out of fold** — in-fold ones "just parrot the labels back".
- The blend was pushed "a lot further toward the model than felt sane"; 50/50 already jumped.

Raymond Yuen describes the same thing with several teachers trained on different weak-label sources. He
says different teachers help different targets.

**Counter-report, same thread (Nicolai Karcher, 09-27):** 50/50 soft bootstrapping gave "a clear lift on
both gold and on (original, non-bootstrapped) labels in local CV but … doesn't translate to LB at all".
Tucker Arrants (08-31): corrected thousands of extraction failures with no LB change.

Chris Deotte (09-25) makes a definitional point. With pseudo-labels, "every single model can match our best
ensemble CV score". That fits our own board read: ensemble-teacher headroom 0.002 (AXES:160).

## Where our record stands on it — mechanism, not number

We have run the same TEACHER, in a different FORM, read on a different RULER:

- **Form.** `--distill-weight 0.7 --teacher-targets teacher_clean.npz` adds `Distillation`
  (`kneemri/training/terms.py:354`). That is a pairwise RANKING term toward the teacher's ordering. It sits
  BESIDE the report-label loss, whose target is untouched. The forum recipe instead REPLACES the BCE target
  with `(1−a)·y_report + a·p_oof`.
  - The difference matters exactly on the silent cells. There the report says 0 while the teacher says
    0.4. The ranking term only reorders pairs; the blended target changes what the main loss fits.
  - `terms.py:357-362` argues ranking beats KL because the metric discards calibration. That argument is
    about the auxiliary term. It does not cover moving the PRIMARY target off a label that under-reports.
- **Ruler.** Every read of that term is on a report-derived holdout: `seat_clean_distill` +0.0056 midpoint
  on the max line; `labelwise_depth24_distill` −0.0010 ledger75, truncated e19, no `.pt` on disk.
  - The ledger itself records that a change moving a model toward annotation truth mechanically LOSES AUC
    against report labels (experiments/README ~3885).
  - It also records that our report labels over-call against the 58 (108 FP vs 32 misses, README ~8322).
  - So the null is on the one ruler that cannot see the claimed mechanism. The gold-58 cannot adjudicate
    either: ~46 of the 58 are in the training folds, and the per-label SE is ~0.12.
  - The un-leaked ruler is the board.

That matches the CLAUDE.md harness list: faulty TEST (a ruler that cannot see the axis) plus an
unfitted FORM. The target-blend form on the shipping line, read on the board, has never run. AXES:154
(relabelling CLOSED) does not cover it. That row is about correcting the report table toward the 58.
This is a model-derived soft target over all 4,349 studies.

**Correction 18:19Z: the form HAS run once, off the shipping line.** `hisrecipe_distill` trained a foreign
`coatnet_rmlp_2_rw_384` on `(1−w)·y + w·teacher_clean_full` (`kneemri/experiments/his_labels_blend.py`),
train rows only.
- Result: ledger75 0.8880 on 896 rows, at the bottom of our band.
- Its correlation with the pool ROSE to ~0.89, above our own members' 0.871
  ([supervision sets correlation](../../interpretations/kneemri/2026-09-08_supervision-not-architecture-sets-correlation.md)).
  Blending toward our own teacher pulls a model onto the pool.
- Also, `report_labels_v2` is already soft (YES 0.82 / UNK 0.28 / NO 0.08). The forum's silent cells are our
  UNK rows, so some of the "keep it fractional" gain is already in the table.

What stays unrun is the form on OUR recipe, read on the board. knee-p0s4f carries the revised prior: ~0.10
solo > 0.946, ~0.02 clears.

**Honest prior.** P(clears +0.015) is low. The same teacher's ranking term is sub-floor, Nicolai reports
no LB transfer at 50/50, and ensemble-teacher headroom on the board is 0.002. It is worth one arm and one
instrument slot because the ruler it was killed on is blind to the axis. It is not worth a sweep.

## The other new axis: a VLM as the image trunk

Qwen 3.5 2B at 384 px, vision encoder plus LoRA on the language model, is claimed at 0.950 single fold in
~3.5 h. Our closed VLM row (AXES:156) is about VLMs as REPORT READERS, never as the image model, so this is
unpriced here.

Taken at face value, the claim is +0.004 over our 0.946. That is below the +0.015 bar, so it is a
diversity or member candidate, not a solo lever by itself. The CNN blind-spot row (AXES:47) sets the
re-open bar for an off-axis member: it has to be within ~0.01 of the champion. A 0.950 claim clears that
bar on paper.

## What the biohub post-mortem transfers

`biohub_kaggle/interpretations/celltrack/converging/2026-09-30_private_lb_post_mortem.md`: seven
submissions spanning 0.945–0.953 public landed in 0.916–0.918 private, a 0.002 spread. Every
late-competition knob was invisible on the private split. The gap to the frontier lived in an axis the
fleet never touched: own models, not tuned forks.

Two transfers, neither a mechanism for knees:

1. Our seat-reshuffle wins inside 0.002 on public are that same flat band. Do not select finals on them.
2. The frontier's gap lives upstream of the knob fleet. Here the thread says upstream means the training
   TARGET, which is the one row above not already measured on a ruler that can see it.

Sources: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304 ;
biohub post-mortem path above; experiments/README.md ~3885, ~8322, ~8564-8620, ~18232;
`kneemri/training/terms.py:354-368`; docs/AXES.md rows 47, 154, 156, 160.

## Addendum 17:20Z — the celltrack session's five transfers, priced against our record

Sent by the biohub session after its private-LB collapse. Each is checked against what we already hold.

1. **Probe the public/private split.** Unpinned here. Our transfer slope is fitted on PUBLIC rows only
   (`board_check`), so it says nothing about private. The test images are hidden, so the scanner/site
   composition of the private slice cannot be measured, only the public fraction read from the rules.
   Selection consequence, zero card: pick finals on the tightest per-stratum holdout spread, not the best
   public point. Our 0.001 ties already bound that spread (AXES:213).
2. **A learned head over member outputs.** Not a solo, so off the current deliverable. The literature
   already says a mean matches stacking once members correlate
   ([stacking_vs_averaging](2026-09-02_stacking_vs_averaging_correlated_members_qag_answer.md)).
3. **Per-label AUC quantization.** Useful arithmetic for choosing which label a target-blend can move: the
   rare labels have the coarsest lattice. Fold it into the read of knee-p0s4f, not a separate run.
4. **Knob with no consumer / type-not-value asserts.** Already guarded here by `run_check`, `mount_check`,
   `witness_check` and the "assert the flag LANDED" rule. knee-p0s4f must assert the blended target
   reaches the loss (the batch's y mean moves off the report mean), not only that `--labels` was passed.
5. **External data pays where supervision is thinnest.** That is the silent-cell target, i.e. knee-p0s4f.
   It is not the trunk.

**Correction 17:22Z, from the same session.**

- *Item 1.* Hidden labels do not hide composition in principle: the kernel sees every test header. What
  blocks it here is the channel back out. A code competition's hidden-test rerun returns only the score,
  never its log or output, and encoding a histogram into submissions is leaderboard probing. So the
  private composition stays unmeasurable **by the rules, not by the data**. The permitted half is the
  editor run over whatever test studies it exposes, read as a public-slice sample only.
- *Item 2.* The covariate head (member outputs plus study metadata) is **PARKED ON SCOPE, not refuted.** The
  stacking note bounds a function of member outputs alone. A per-stratum reweighting driven by a covariate
  is a different object, and nothing here measured it. It re-opens if the deliverable stops being a solo,
  and a real scanner stratum ([scanner axis](../../interpretations/kneemri/2026-09-05_the-scanner-axis-exists-in-the-data-and-not-in-the-ruler.md)) is its gate.

**Settled 17:27Z.** The celltrack session withdrew its item 1. Its own split was visible in the data
description, not read from headers. What stands:

- **The selection rule carries the whole weight.** Pick finals on the tightest per-stratum holdout spread, not
  the best public point. It is the only lever left against an unknown private composition, not a nicety.
- **Known and unresolvable:** a private stratum that is ABSENT from train and holdout entirely.
  Normalization cannot remove an axis it never saw, and the rules forbid measuring one. It is stated here so
  nobody re-derives the chain.
- **Not priced:** standardizing each study on header covariates (field strength, TR/TE, spacing, thickness)
  inside the kernel. That is legal, since the kernel only reads its own input. Before anyone builds it, check
  what the preprocessing already normalizes.
- **Item 2's gate is unmeasurable**, so the covariate head re-opens on a SCOPE change only, never on a
  measurement arriving.
