#!/usr/bin/env bash
# stack_increments_base.sh — the two increments again, on a base that is NOT the calibrated corner.
#
# WHY THIS ARM EXISTS. `knee-orjq` closed stacking: seven attempts against `calib_dense_hold` 0.8634, all
# seven below it, p = 2^-7 = 0.008, monotone in perturbation size. That close preserves exactly one limit
# in writing — A STACK BUILT FROM A DIFFERENT BASE HAS NEVER BEEN TRIED, and the closure is a result about
# the calibrated corner specifically. This arm is that test and nothing else.
#
# WHAT MOVED, AND IT IS ONLY THE BASE. `stack_increments` ran the two terms below on top of
# --calibrated-targets --hold-annotations and read 0.8642 against that base's 0.8634: +0.0008, NULL inside
# the 0.0062 floor. Here the SAME two terms run with those two flags REMOVED. Every other flag is
# byte-identical to `stack_increments`, so a difference between the two runs is attributable to the base
# and to nothing else.
#
# THE TWO TERMS are unchanged and are quoted with their own controls, not re-measured here:
#   1. LABEL TABLE — blend_three_v1.csv (three independent readers, MEAN mode) read 0.8516 against
#      holdannot 0.8482, +0.0034. --eval-labels report_labels_v2.csv is pinned so the RULER does not move.
#   2. DISTILLATION — --distill-weight 0.3 with raptor_teacher.npz read 0.8613 against band_dense 0.8578,
#      +0.0035. Teacher targets are consumed on TRAIN studies only, so this is not knee-ie30's leak.
#
# THE MECHANISM THE NULL SUGGESTS. Calibration and the blended table both move the TARGET LEVEL. If they
# are one increment wearing two names — which is what `stack_increments`' null says on the reading knee-24bx
# already recorded — then removing calibration should let the label term arrive at its own measured size
# instead of landing on headroom already spent. That prediction is falsifiable in one run.
#
# control  band_dense = 0.8578 @ 336/20, the plain base with neither term. Same 909-study holdout, same
#          eval labels. NOT calib_dense_hold: this arm is not built on it and must not be read against it.
# cost     12 epochs at 336 px, ~362 s an epoch measured on contrast_pool = ~72 min.
#
# PRE-REGISTERED against band_dense 0.8578. Floor 0.0062, board resolution 0.0015.
#   ADDITIVE — solo at or above 0.8640 (+0.0062). The terms compose off the calibrated corner, knee-orjq's
#     preserved limit resolves in favour of the base being the cause, and the stack thesis is live again on
#     the plain lane. Next move is the 224 geometry decomposition, not a slot.
#   PARTIAL — 0.8613 to 0.8639. One term's worth arrived. Same reading as before: not resolvable alone.
#   NULL — 0.8516 to 0.8612. The terms do not compose on ANY base tried, which promotes knee-orjq's close
#     from a statement about the calibrated corner to a statement about the terms, and closes the last
#     preserved clause in it. That is a real finding and it retires the branch.
#   REGRESSION — below 0.8516, i.e. under the weaker of the two terms' own solo results. Then the terms are
#     antagonists on the plain base while merely flat on the calibrated one, and the suspect named in
#     stack_increments applies with more force: the teacher's ordering was fitted against a different label
#     table than the student now trains on.
#
# bead: knee-0wta
#
# deliberate: --labels        (blend_three_v1.csv, term 1)
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

run_train stack_increments_base \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --distill-weight 0.3 --teacher-targets "$PROCESSED/raptor_teacher.npz" \
  --labels "$BLEND" --eval-labels "$LABELS" --out "$MODELS/stack_increments_base_s2026.pt"
