#!/usr/bin/env bash
# stack_increments.sh — two increments this project measured separately and never composed.
#
# WHY THIS ARM EXISTS, AND WHY TODAY. `board_transfer` was refit on the completed record 2026-08-30 and
# reads +0.98 board points per holdout point among entries at or above 0.8670 (16 resolved pairs,
# [+0.94, +1.08]), with 0/252 inversions among the pairs the holdout resolves at all. Offline gains at
# the top of our record transfer to the board ONE-FOR-ONE. The separate 23% figure this repo quotes is
# SOLO into SEAT — a member's own gain diluted by the ensemble it joins — not offline into board, and
# welding the two produced a sentence that said offline work was worthless. It is not. See knee-24bx.
#
# WHAT FOLLOWS FROM THAT. The project holds a pile of increments that are mechanism-sound, reproduced
# against their own controls, and individually under the 0.0062 paired floor, so each was filed as "not
# a finding" and set down. NONE OF THEM HAS EVER BEEN STACKED. CLAUDE.md is explicit that increments
# below the bar still compose and that a chain summing past it IS a result; this is the first arm to
# test whether these particular ones do.
#
# THE TWO TERMS, each with its own measured delta and its own control:
#
#   1. LABEL TABLE. `blendlabels` trained on blend_three_v1.csv — the table blended from three
#      independent readers of the same reports (core.data.labels.table_blend, MEAN mode) — read 0.8516
#      against holdannot 0.8482, +0.0034. `--eval-labels report_labels_v2.csv` is pinned so the RULER
#      does not move with the arm; that pin is not optional and its absence makes the run unreadable.
#   2. DISTILLATION. `raptor_distill` at --distill-weight 0.3 with raptor_teacher.npz read 0.8613
#      against band_dense 0.8578, +0.0035. The teacher is foreign and its ordering is partly learnable.
#      This is NOT the contamination knee-ie30 found: teacher targets are consumed on TRAIN studies
#      only, so a teacher that saw our holdout cannot leak a holdout label through them.
#
# BASE RECIPE is `calib_dense_hold` — 0.8634, the best offline single member this project has produced,
# and the control this arm is read against. Its two flags (--calibrated-targets --hold-annotations) stay
# exactly as it set them, so the ONLY moved variables are the two terms above.
#
# ADDITIVITY IS THE HYPOTHESIS, NOT AN ASSUMPTION BEING SMUGGLED. 0.8634 + 0.0034 + 0.0035 = 0.8703,
# a delta of +0.0069 over the control, which CLEARS the 0.0062 floor. That is the point of stacking two
# rather than testing one: it is the first version of this question our own ruler can resolve.
#
# THE CAVEAT ON THE TERMS, stated because it is the likeliest way this reads null. blendlabels measured
# its +0.0034 at 224 px on 12 slices against a control that does not hold annotations; this arm runs at
# 336 px on 20 slices holding them. Transporting a delta across an operating point is an assumption, and
# a null here does not distinguish "the terms are not additive" from "the label term does not survive
# the move to 336/20". If it reads null, the decomposition is one arm at 224 matching blendlabels'
# geometry with distillation added — cheap, ~20 min — before the stack thesis is touched.
#
# control  calib_dense_hold = 0.8634 @ e6, same recipe minus the two moved flags, same 909-study holdout.
#          band_dense = 0.8578 and holdannot = 0.8482 are the two terms' OWN controls, quoted above only
#          to source the deltas; neither is this arm's comparison.
# seat     priced with record_mix against comp/members.roster afterwards, card-free. Note the 0.23
#          solo->seat dilution before reading the seat number as a disappointment: +0.0069 solo is
#          ~+0.0016 in a single seat, and the ensemble move this arm is really aiming at comes from
#          retraining the SEATS on a landed recipe (calibsix_v1's precedent), not from seating this one.
# cost     12 epochs at 336 px, ~362 s an epoch measured on contrast_pool = ~72 min.
#
# PRE-REGISTERED. Floor 0.0062, board resolution 0.0015.
#   ADDITIVE — solo at or above 0.8696 (+0.0062 over the control, the floor). The two terms compose,
#     the stack thesis is live, and the next move is NOT another single-seat submission: it is retraining
#     the champion's six seats on this recipe, which is the move whose gain transfers one-for-one.
#   PARTIAL — solo 0.8669 to 0.8695. One term's worth arrived and the other did not, or both arrived
#     at half size. Not resolvable alone; the reading is the 224 decomposition named above, not a slot.
#   NULL — solo 0.8600 to 0.8668, i.e. inside the floor of the control. The increments are not additive:
#     they are one increment wearing two names, which is a REAL finding and the same shape as the pool
#     being 82 percent one number. It would close the "bank the increments" branch of knee-m69u and
#     leave the non-DINOv2 lane as the only one standing.
#   REGRESSION — solo below 0.8600. The two terms are ANTAGONISTS, which no reading currently predicts.
#     Distillation moves ordering and the blended table moves the target level; if they fight, the
#     mechanism to look at first is that the student is being asked to match a foreign ordering fitted
#     against a DIFFERENT label table than the one it now trains on. That is a real and specific
#     suspect, and it would be tested by distilling against a teacher re-scored on blend_three_v1.
#
# bead: knee-24bx
#
# deliberate: --labels        (blend_three_v1.csv, term 1 — the whole point of the arm)
# deliberate: --eval-labels   (report_labels_v2.csv — the ruler must NOT move with the arm)
# deliberate: --distill-weight
# deliberate: --teacher-targets
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

BLEND="$PROCESSED/labels/blend_three_v1.csv"
require "$LABELS"
require "$BLEND"
require "$PROCESSED/raptor_teacher.npz"

run_train stack_increments \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --distill-weight 0.3 --teacher-targets "$PROCESSED/raptor_teacher.npz" \
  --labels "$BLEND" --eval-labels "$LABELS" --out "$PROCESSED/stack_increments_s2026.pt"
