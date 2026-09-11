#!/usr/bin/env bash
# The tear-column wholesale correction, on the card. Same recipe as `band_span` and `v5blend_span` in
# every other respect; the only substitution is `--labels`.
#
# THE TABLE. `report_labels_v6tears.csv` is v5blend with the cluster's soft value written into the 1,379
# CONTESTED tear cells of `v2_lone_dissent.csv` (ACL 298, MCL 83, Medial Meniscus 567, Lateral Meniscus
# 431; 1,097 studies; 1,377 of them v2=1 over-calls). `research/scripts/build_v6_tears.py` builds it and
# names the operation: `column-substitution`, second use - NOT a blend, since `table_blend.py` is a
# mean/tiebreak combiner over foreign readers of the same report (D348).
#
# WHY THIS AND NOT ANOTHER KNOB. 469 holdout dissent cells were adjudicated blind, ten chunks, the key
# unopened until the last verdict was on disk. 277 of them are tear-column cells. On all 277 v2 is WRONG
# and the cluster is RIGHT: accuracy 0.000 against 1.000, over 223 distinct studies. The draw carries no
# label-side variable - fold assignment is a hash of the REPORT TEXT (`core/data/corpus/folds.py:61`) -
# and the 469 holds 20.1% of the corpus's tear-dissent cells against the holdout's 20.6% share of
# studies, so the rate is a population statement, not a sample artefact (D350).
#
# THE TWO GUARDS, STATED BEFORE THE RUN.
#   1. Only the selector's contested cells are written. v2's UNCONTESTED tear cells are a different
#      population, where the first audit put unanimous-cell error at 1.48%. The agreed mass is untouched.
#   2. The 1.000 rests on 223 studies, so a residual cluster error at the ~1% level is invisible at that
#      n: about 14 wrong cells across the 1,097 studies. That is the entire residual-risk budget.
#
# THE PRICE, BEFORE THE RUN. 12 epochs, ~35 min on the 5090 at the 173 s/epoch `v5blend_span` is
# measuring. It corrects 1,379 of roughly 36,000 tear cells - 3.8% of the four columns, ~1.3% of the
# table - and every one of them is a false POSITIVE being removed from the class the tear heads are
# worst at. Plausible size if the mechanism carries: +0.01 to +0.02 on the four tear columns, a quarter
# of the twelve-column mean, so +0.003 to +0.005 overall on the v2 ruler's own terms and more on the
# corrected T1. Odds it clears the 1.5% bar on the twelve-column mean ALONE: low. It is queued ahead of
# `labelwise_depth24` anyway because that arm's own pre-registration predicts flat, and this one is the
# largest label correction the project has measured rather than argued.
#
# THE RULER, AND THE SAME TRAP. `--eval-labels` stays pinned to v2, exactly as `v5blend_span` does, so
# the ledger number is comparable to `band_span` 0.8565 midpoint / 0.8592 T1. A member trained on a table
# whose errors it was taught NOT to reproduce reads a FLOOR on the v2 ruler by construction. The read is
# the corrected T1 rescore off this run's holdout npz.
#
# PREDICTION, written before the run. On the v2 ruler: at or below `v5blend_span`, and the tear columns
# specifically should fall - the ruler still holds the 1,377 over-calls this table removed. On the
# corrected T1 ruler: at or above `v5blend_span`. T1 below `v5blend_span` refutes the substitution and
# sends the 277/277 back to the harness, PLUMBING first.
#
# Assert the recipe's `labels` field names v6tears and its `eval_labels` names v2 before writing any
# number here down.
#
# reading: holdout909
# bead: knee-57aec
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

V6="$PROCESSED/labels/report_labels_v6tears.csv"
require "$V6"
require "$LABELS"

run_train v6tears_span \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.1 --band-high 0.9 \
  --labels "$V6" --eval-labels "$LABELS" --out "$MODELS/v6tears_span_s2026.pt"
