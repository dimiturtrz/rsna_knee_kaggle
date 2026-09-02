#!/usr/bin/env bash
# bead: knee-5pdl
# band_dense's geometry with the five reader levels read as an ORDERED LADDER instead of five values.
#
# WHAT THE THREE PREVIOUS SEVERITY ARMS ALL SHARED, and it is not the thing any of them varied.
# The reader emits 0.08 / 0.28 / 0.68 / 0.82 / 0.94 and the plain loss asks ONE logit per label to LAND
# ON each of those numbers. calib_span moved the numbers (-0.0020), severity_weight moved the weights
# (-0.0048), severity_flat removed the gradation entirely (-0.0044). Every one of them kept the
# single-logit value-fitting parameterisation, so the grade distinctions and the positive/negative
# separation the AUC actually scores were still competing for the same monotone map.
#
# WHAT CHANGES. P(level >= k) = sigmoid(z - t_k) for the four boundaries between adjacent levels. A
# severe study only has to OUTRANK threshold 4 rather than arrive at 0.94 — strictly weaker, and
# exactly as informative, because order is all an AUC reads. The four thresholds are derived from the
# measured agreement rates (0.041 / 0.136 / 0.378 / 0.688 / 0.958, logit-space midpoints), so they are
# ordered by construction and nothing is tuned.
#
# NOTHING DOWNSTREAM CHANGES. The thresholds are subtracted inside the loss, so z stays the single
# per-label logit the holdout ranks, the checkpoint stores and the kernel serves. This arm can only be
# a loss result; there is no inference path for it to be confounded by. Gradient scale was measured
# before queueing under a realistic level mix (45/25/12/13/5, batch 64): plain 0.0124, ladder 0.0156 —
# 1.26x, same order, so no learning-rate confound wearing a null's clothes.
#
# control  band_dense = 0.8578 at epoch 8, matched geometry, matched recipe, matched epochs.
# cost     ~55 min of card, identical step time (the spread is four cat()s on a (b,12) tensor).
#
# WHAT IT IS PRE-REGISTERED AGAINST, AND IT IS A SEAT AND NOT A SOLO NUMBER. Ten axes have now closed
# inside the 0.0062 noise floor, and the champion's whole +0.011 over its best member came from
# ensembling. Every member in the roster optimises the SAME binary target and differs only in slice
# geometry, so their errors correlate through the objective. This is the first member optimising a
# DIFFERENT target on identical inputs. PASS = >= +0.0015 into the roster of five; solo AUC decides
# nothing either way. FAIL is the strong outcome: diversity is then saturated on every reachable axis
# and the 0.054 frontier gap is not an ensembling gap.
#
# THE MECHANISM GUARD. severity_weight lost by starving the batch — its weighting left few effective
# positives per step. This cannot reproduce that: every cell contributes four crossings at its OWN
# weight (repeated, not divided), so the effective supervision per step goes UP, not down. If this arm
# reads like severity_weight, the ladder is not why.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train grade_ladder \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --grade-ladder \
  --labels "$LABELS" --out "$MODELS/grade_ladder_s2026.pt"
