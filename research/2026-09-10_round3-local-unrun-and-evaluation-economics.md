# 2026-09-10 — Round three: the repo's own unrun machinery + the evaluation-economics literature

Owner-directed third sweep, done by the lane directly (not delegated): (i) what is BUILT but
never run in this repo, and (ii) the online literature on the two questions main's letters
keep circling — how much a bigger gold set buys, and whether board reuse corrupts our ranking.
Provenance: letters D303–D308; mapping file 2026-09-10_unseen-candidates-mapped-onto-our-
measurements.md. Not committed by this lane.

## Part 1 — local: built but unrun (rough inventory)

Method: 317 scripts in experiments/arms/ name-matched against metrics/checkpoints on disk —
**~76 appear unrun by name-match**; the method over-counts (arm names ≠ --out names;
oof_band_folds flagged as unrun yet landed null yesterday), so treat the list as a review
queue, not a census. The interesting ones, with their own header texts:

- **`ortho_fresh_lowlr.sh`** — the named control for ortholarge "probes best, serves worst"
  (mapping file, row 5). LP-then-unfreeze vs dense-fit differ in TWO variables; this arm
  isolates them. Directly gates the best-supported discovery candidate (frozen in-domain
  FM features). Unrun.
- **`multiscale.sh`** — reads every sequence TWICE (130 mm context + 80 mm detail) with slot
  attention pooling over scale — the within-slice/extent-composition idea that survived
  M278–M280 (extent flat, padding innocent, scale matched) without ever being tested as a
  TWO-extent model. Built to the spec of bead knee-1nn; gated on crop80_tissue.
- **`sided_tissue.sh`** — `sided_pool` with the crop about TISSUE instead of the image; the
  suspect its own ledger row named and did not clear. One flag.
- **`slice_count_whole_stack.sh`** — 12 cached slices → 16 read whole-stack; the header
  documents why the earlier 20-vs-12 null (0.8309 vs 0.8310) could not see the mechanism
  (both read ONE random 3-slice window per step).
- **`fov_perstudy_read.sh`** — a READOUT arm, no training: the foreign readout threw away
  per-study probabilities, so the eleven-member toll table has no subsets in it. This arm
  keeps them. Pure CPU-side analysis shape.
- **`diversity_curve.sh`** — offline measurement: does mechanism-ensembling buy what
  fold-ensembling bought, on members already paid for. No card.
- **`latent_rate_ladder.sh`** — the latent-target pretrain lr ladder (7 min a rung),
  bead knee-lm37v.
- **`report_supervision.sh`** — built and REFUTED in its own header (report embedding
  cosine supervision); do not revive without a new argument.

Training-side machinery in kneemri/training/ that exists as modules: `sampling.py`
(Disagreed + Ramped samplers — the curriculum family's mechanism), `feature_distillation.py`,
`pretrain.py`/`spectral_mae.py` (the frequency-domain objective, flagged in the mapping as
the cheapest open cell). Unfreeze-depth ladder arm `unfreeze_depth.sh` also sits unrun.

## Part 2 — online: evaluation economics (the questions behind "more gold" and the board)

**Does leaderboard reuse corrupt rankings? Mostly no, by the meta-analysis.**
- Roelofs et al., NeurIPS 2019, "A Meta-Analysis of Overfitting in Machine Learning":
  over 100+ Kaggle competitions, **no widespread evidence of test-set overfitting** was
  found despite heavy adaptive reuse. (papers.nips.cc/paper/9175)
- Mania et al., NeurIPS 2019, "Model Similarity Mitigates Test Set Overuse": adaptive reuse
  corrupts much less than worst-case theory predicts when the submitted models are similar
  to each other — exactly the regime of a mature leaderboard where everyone iterates
  near-identical recipes.
- Blum & Hardt 2015 "The Ladder" is the theory baseline (competition-grade holdout design);
  the empirical answer is that plain public leaderboards survive reuse far better than the
  theory's worst case.
- Recht et al. 2019 (CIFAR-10 re-do) shows ranking *largely preserved* across a fresh test
  set for methods far apart, with small-model orderings tightening near the frontier.
  Our own external-cohort file (2026-09-10_external-cohort-ranking-does-not-reliably-survive)
  is the counterpoint: in medical imaging the flips DO exist (Klontzas, NCCID) — the
  difference is cohort shift, not reuse. Both effects need to be in the same frame.

**How much does a bigger gold set buy? Diminishing returns past ~1–2k labels.**
- EACL 2026 industry track (arXiv 2512.08121): sweeping golden-set size 25 → 51,000 at fixed
  eval set: large ranking-loss gains up to **~1,000–2,000 labels, then a plateau**. Our
  gold-58 sits deep in the steep part of that curve — the direction "more gold" is
  literature-supported, and the literature also names the ceiling: the next log-decade of
  gold is worth far less than the first.
- Gallas 2006 (one-shot MRMC variance for reader-averaged AUC): the classical machinery for
  pricing how much of a gold-set read's variance is readers vs cases — the formula exists
  if the adjudicated-gold economics ever need a variance decomposition.

**Reader-disagreement as label noise** — Ju et al. 2022 (PMID 34995185, 181 cites):
systematic separation of disagreement noise (expert-vs-expert) from single-expert noise on
medical images with dual-uncertainty estimation — the nearest published frame to our
MEASURED_AGREEMENT ladder, in case a reader-weighted training target is ever built.

**Physics-based MRI augmentation for classification: SILENT.** Motion/artifact simulation
work (Safari 2025 meta-analysis; Eichhorn dissertation; AAPM 2026 dual-branch) is all
reconstruction/detection endpoints — nobody reports a disease-classification delta from
physics-simulated MRI augmentation. This is an unpriced cell adjacent to our closed
augmentation family, distinct from it (k-space-level perturbation, not pixel-level).

## What round three changes

- Locally: the two cheapest unrun cells are now on record with headers quoted —
  `fov_perstudy_read` (no card) and `diversity_curve` (no card), then `ortho_fresh_lowlr`
  and `multiscale` (card). Everything else in the unrun list is either refuted-in-header,
  a teardown/smoke utility, or era-confounded.
- Online: the board-reuse worry gets weaker (meta-analysis + similarity argument), the
  more-gold direction gets a *bounded* yes (steep until ~1–2k, plateau after), and the
  physics-augmentation cell is named as silent rather than tried-and-failed.
