#!/usr/bin/env bash
# oof_teacher.sh — the one distillation form never tried: OUR pool's own out-of-fold posterior as teacher.
#
# WHAT THE THREE FLAT DISTILL ARMS SHARE, AND WHY THIS IS NOT A FOURTH OF THEM. `foreign_distill` (0.3),
# `foreign_distill_mix07` (0.7) and `dissent_seat` all taught from `teacher_foreign_corpus` — a FOREIGN
# pool whose ordering the seat curve places INSIDE our own consensus. Each increment of teacher weight
# therefore pulled the student toward the pool it was supposed to diverge from, which is exactly what the
# js divergence recorded (0.0273 no-teacher -> 0.0190 at weight 0.3). Those three close the FOREIGN
# teacher, not distillation. This arm changes the teacher and holds everything else, so it is one
# variable wide against `foreign_distill_mix07`.
#
# WHAT THE NEW TEACHER IS. `kneemri.evaluation.oof_teacher` concatenates `oof_fold1..4_s2026.holdout.npz`
# — 875 + 884 + 849 + 890 = 3498 studies, the whole TRAINING half, each study read by the one fold model
# that did not train on it. Fold 0 is absent and must be: it is the holdout every arm is scored on, so a
# teacher covering it would leak the evaluation labels through this file. 3498 is the complete
# denominator, not a gap in a 4407 one.
#
# THE MECHANISM IT ADDRESSES. Our label table is five discrete levels; `label_disagreement` prices ~20%
# of read cells wrong and 38.5% of cell PAIRS tied. Every label treatment so far reached a fraction of
# that: `blendlabels` (+0.0034) breaks ties by a three-reader mean, `errors_relabel` reaches the cells a
# regex can find. A continuous out-of-fold posterior orders EVERY cell, which is the only treatment whose
# denominator is the whole defect rather than a slice of it.
#
# ONE VARIABLE against `foreign_distill_mix07`: --teacher-targets. Weight held at 0.7 (the best solo point
# on the monotone line), rate, slices, band, pooling, whole-stack and epochs all held.
#
# HOW IT IS READ, written before the number (pre-registered with mastermind, 20:23Z).
#   * v2 ruler >= 0.8640 AND gold above blendhold's 0.8952 — the corpus teacher is the label lever, and
#     tomorrow's card is a member factory on it (five folds, then a seat).
#   * v2 flat but gold up — same conclusion, and the v2 ruler is the thing that was wrong.
#   * both flat — distillation closes on four points, and the remaining label lever is a better READER
#     (an 85-level relabel of the wrong cells), not a mean blend.
#
# COST. ~70 min of card, 12 epochs on the shared recipe. ZERO submission slots. Teacher file already on
# disk — it is assembled from readings that exist, so it cost no card of its own.
#
# deliberate: --teacher-targets (ours out-of-fold, the single moving variable)
# deliberate: --distill-weight (0.7, HELD from foreign_distill_mix07)
# bead: knee-bq62a
# reading: oof_teacher_s2026.holdout.npz
# control  band_dense = 0.8578
# The second control is `foreign_distill_mix07` — same distill weight, foreign teacher — which is what
# separates "a teacher helps" from "OUR out-of-fold teacher helps". `arm_read` parses one control line
# only, so the no-teacher solo is the one named in machine form; the teacher control is read by hand.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
require "$PROCESSED/teacher_oof_corpus.npz"

run_train oof_teacher \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --distill-weight 0.7 --teacher-targets "$PROCESSED/teacher_oof_corpus.npz" \
  --labels "$LABELS" --out "$MODELS/oof_teacher_s2026.pt"
