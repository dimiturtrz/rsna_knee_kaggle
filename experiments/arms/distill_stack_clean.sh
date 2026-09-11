#!/usr/bin/env bash
# distill_stack_clean.sh -- distill_stack_max with ONE variable moved: the teacher line. Foreign corpus
# out, our own CLEAN out-of-fold teacher in.
#
# bead: knee-ia49g
# reading: distill_stack_clean_s2026.holdout.npz
#
# WHY THIS AND NOT ANOTHER CELL. D375 priced the two open levers side by side and the seat lever lost:
# greedy seat search survives the half-split at +0.0013 +/- 0.0013 (37/40 sign, k=8) with selection
# Jaccard 0.22, i.e. about one SD of holdout noise and no seating worth naming in a kernel. The member
# lever is the only one whose size matches the hole. So the card stays on members.
#
# THE ONE VARIABLE. Against `distill_stack_max` (on the card now): --teacher-targets moves from
# teacher_foreign_corpus.npz to teacher_clean_full.npz. Everything else is held byte-for-byte -- epochs,
# labelwise pooling, whole-stack, lr 4e-5, 20 slices, band 0.1/0.9, calibrated targets, hold
# annotations, distill weight 0.7, blend_three labels, v2 eval labels, batch 6 accumulate 2.
#
# TWO THINGS IT BUYS, AND THE SECOND IS THE ONE I EXPECT TO CASH.
#   1. Attribution on the teacher line. distill_stack_max stacks five marginals at once and deliberately
#      claims no attribution. This arm cannot separate the other four either, but it DOES separate the
#      one term the whole distill family rests on: whether the +0.0151 distill marginal @honest302 is
#      about foreign supervision specifically or about teacher supervision generally.
#   2. Ensemble diversity, which D373 says is the unspent lever. 45,451 member pairs have mean pairwise
#      Spearman +0.0144 on column-centred profiles with within-member split-half reliability +0.4965 --
#      the plateau is 303 functions failing on different columns, and ensemble headroom implied by that
#      was never spent by the solo plateau. A member trained on a DIFFERENT teacher line is the cheapest
#      honest source of a decorrelated function this project has. Its solo number does not have to beat
#      anything for it to earn a seat.
#
# THE CEILING THIS IS READ AGAINST MOVED TONIGHT. oof_teacher_s2026's 0.8685 is void: oof_teacher.sh:64
# and seat_clean_distill.sh:45 are identical run_train blocks apart from --teacher-targets (leaky
# teacher_oof_corpus vs clean teacher_clean), and they read 0.8685 vs 0.8634 @holdout909 midpoint. The
# fold-0 leak transfers at +0.0051, which is the whole margin by which that member led. **The honest
# pool max is 0.8649 (clean_calib_s2026).**
#
# HARNESS CHECK BEFORE READING THE NUMBER. The printed recipe must name blend_three_v1.csv on "labels",
# report_labels_v2.csv on "eval_labels", "slice_pooling": "labelwise", a non-zero distill weight with
# teacher_clean_full.npz, calibrated_targets true, hold_annotations true. Three runs have entered this
# ledger as facts about an architecture when they were facts about a flag that never landed.
#
# VRAM. Identical to distill_stack_max, which reserved 13.11 GB of 32.6 at --batch 6 --accumulate 2 on
# the canonical six-channel scheme. If the launch line prints a cap above 31.8 GB, kill it rather than
# let Windows page over PCIe at 11x.
#
# PRE-REGISTERED @holdout909 midpoint, against the honest ceiling 0.8649 and against distill_stack_max's
# own number (call it S, read when it lands):
#   >= 0.8650 AND within 0.002 of S   the stack cell is real and the teacher LINE is not what carries
#             it. Both members are seat candidates and the pair is the diversity buy. Best case.
#   >= 0.8650 AND more than 0.002 below S   the stack is real and FOREIGN supervision is the term. The
#             next card is more foreign teachers, not more stacking.
#   >= 0.8650 AND more than 0.002 above S   the foreign teacher was the WEAKER line all along and every
#             foreign-distill member in the pool is under-read. That would re-open D371's cell (d).
#   <= 0.8649 at or below the honest ceiling. Then this teacher line adds nothing to the stack and the
#             member is judged on its ensemble marginal into honestsix alone, not on its solo.
#
# PRICE: ~1.5 h of card, ZERO submission slots. Teacher and label table both on disk. Expected solo
# 0.865-0.874 by analogy with its foreign twin; expected ensemble value is the reason it runs.
#
# deliberate: --teacher-targets (teacher_clean_full.npz, the single moving variable vs distill_stack_max)
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

BLEND_THREE="$PROCESSED/labels/blend_three_v1.csv"
TEACHER="$PROCESSED/teacher_clean_full.npz"
require "$LABELS"
require "$BLEND_THREE"
require "$TEACHER"

run_train distill_stack_clean \
  $RECIPE --epochs 12 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --hold-annotations \
  --batch 6 --accumulate 2 \
  --distill-weight 0.7 --teacher-targets "$TEACHER" \
  --labels "$BLEND_THREE" --eval-labels "$LABELS" --out "$MODELS/distill_stack_clean_s2026.pt"
