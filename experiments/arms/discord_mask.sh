#!/usr/bin/env bash
# discord_mask — mask the DISAGREEMENT-SELECTED minority of label cells, keep the rest untouched.
#
# WHY. The 402-cell blind adjudication (2026-09-11) measured extractor error in two strata, against a
# key I never opened, under a rubric fixed before the first verdict:
#     unanimous cells   5/338 = 1.48%  (u95 2.37%)
#     discordant cells 10/64  = 15.6%  (u95 20.3%)
# Ten and a half times. The selector is four-way extractor disagreement over blend_four,
# blend_hybrid_fx_v4, confident and report_labels_v3 binarised >=0.5 — already computed, free, and
# model-free. It flags 7369 of 52872 cells (13.94%).
#
# The published shape says repair the flagged minority and LEAVE THE REST ALONE: FCF (Jiang, AAAI-24)
# Thm 6 — low-noise-probability cells "should be retained without any change"; Thm 5 — correction beats
# filtering at equal cleaned-noise rate. Re-extraction is not built yet, so this arm takes the cheaper
# rung of the same ladder that DivideMix (ICLR 2020) ranks above discarding: mask the CELL and let it
# be unlabeled. Multi-label, so the cell goes UNK, never the study.
#
# NOT A REPEAT of confident_span. Measured, before arming: the discordant cells and confident.csv's UNK
# cells overlap on 188 cells — 2.55% of discordant, 1.3% of UNK. Disjoint populations. confident_span
# clears low-CONFIDENCE cells (27.3%, extractor-self-reported); this clears DISAGREED cells (13.9%,
# cross-extractor, adjudicated at 15.6% true error). One is a threshold, the other is a measurement.
#
# control  band_span = 0.8565, same recipe, same fold, same span, same density, --labels report_labels_v2.
#          confident_span (running) is the second control: same idea, different selector.
# cost     band_span's step time, 12 epochs, ~1 h.
#
# PREDICTION, written before the run. Masking 13.9% of cells removes supervision measured at 15.6%
# error and keeps supervision measured at 1.48%. If label noise is what bounds this encoder, this is the
# cheapest test of it that exists — the selector is free and the cells are the ones that are actually wrong.
#   >= 0.8627   the discordant stratum was costing us score; selective re-extraction is worth building,
#               and the repair route stops being a hypothesis.
#   0.8503-0.8627  a null at a floor: removing the worst-measured 13.9% of supervision neither helps nor
#               hurts, which prices the whole label axis as flat in this range and is itself a finding.
#   <= 0.8503   the discordant cells carry usable signal DESPITE 15.6% error — consistent with Menon 2015
#               (one-sided noise preserves the Bayes-optimal ranking), and the answer is to CORRECT them,
#               never to drop them. That sends the route to re-extraction rather than killing it.
# Anticorrelation with confident_span would be the informative outcome: same axis, disjoint selectors.
#
# deliberate: --eval-labels is PINNED to the control's table (report_labels_v2). discordant_masked.csv
#   scores holdout cells differently, and without the pin this comparison would be a ruler change wearing
#   a label-content result's clothes. Same pin, same reason, as confident_span. (knee-df7t)
# reading: D:/models/rsna_knee_abnormality/experiments/discord_mask_s2026.metrics.json
# bead: knee-nz6r7
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

MASKED="$PROCESSED/labels/discordant_masked.csv"
require "$MASKED"
require "$LABELS"

run_train discord_mask \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.1 --band-high 0.9 \
  --labels "$MASKED" --eval-labels "$LABELS" \
  --out "$MODELS/discord_mask_s2026.pt"
