#!/usr/bin/env bash
# stack_increments_drop40_long.sh — knee-p29bb, the gating instrument of knee-289qc.
#
# WHAT THIS IS FOR, AND IT IS NOT ITS OWN SCORE. `stack_increments_ff` is a seat in the champion. Its
# fold-0 twin is `stack_increments_s2026` — the same recipe minus `--fold -1 --ema`, nothing else — and
# it reads 0.8931 ledger75 PEAKING AT EPOCH 9 OF 12 AND THEN DECLINING. So does every other champion
# seat's twin (ledger75, 2026-09-13):
#
#   stack_increments_base  0.8939 @e9    grade_ladder   0.8868 @e6
#   stack_increments       0.8931 @e9    through_dense  0.8867 @e8
#   calib_span             0.8895 @e9    lr4e-5_ws      0.8840 @e8
#
# A budget wall peaks on the LAST epoch. Peaking at 6-9 of 12 and declining for three to six is a missing
# regulariser, and it is the SAME diagnosis that sent the operator pair onto the labelwise family — where
# it then cleared +0.0040 on five of five bases tried (depth24 +0.0055, ext +0.0059, dense +0.0084,
# chainfour +0.0070, blendhold +0.0050). knee-fix1t says the 0.944 ceiling is the POOL and not the
# seating; this pair is the only operator this project owns that lifts the pool, and the champion's own
# six have never been offered it.
#
# WHY A FOLD-0 TWIN AND NOT THE CHAMPION SEAT DIRECTLY. A `--fold -1` refit holds out nothing and no
# ledger row can ever price it — `rampsix` spent 4.2 h of card on a gate unreadable by construction
# (knee-s8kim). Refitting all six champion seats blind would be ~12 h of card against a number that could
# only ever be read on the board, one slot at a time. So the cheap measurable proxy gates the expensive
# commit: this one arm, on the twin, decides whether those twelve hours are spent.
#
# ONE CHANGE, TWO FLAGS. Every flag below is `stack_increments.sh`'s, byte for byte, plus
# `--slot-dropout 0.40` and `--epochs 12 -> 24`. 0.40 is the operating point legislated in knee-yotj8,
# not a swept knob. Expect exactly `slot_dropout 0.0->0.4`, `epochs 12->24` and `out` in the
# recorded-recipe diff.
#
# PRE-REGISTERED, before the run, against its own base at 0.8931 ledger75:
#   >= +0.0040   the pair transfers to the champion's recipe family. The five remaining full-fit refits
#                are licensed — ~12 h of card, spent blind on THIS number.
#   +0.0015..+0.0039  partial. Refit only the two champion seats whose twins peak EARLIEST, where the
#                missing-regulariser diagnosis is strongest: grade_ladder (@e6) and through_dense (@e8).
#   <= +0.0014   the pair does not reach this family. knee-289qc CLOSES, which retires the largest
#                mechanism-backed projection this project holds and is worth the card on its own.
#   solo floor   >= 0.8931, or the transfer failed HERE and no branch above may be read.
#   peak         if this peaks on its LAST epoch, read `convergence_check` (knee-a21h9) — argmax-last AND
#                a final-three-epoch gain over the 0.0062 holdout floor is what refuses a verdict, not
#                argmax-last alone.
#
# PRICE: ~1.6 h. P(clears +0.0040) ~ 0.75 — SEVEN of seven bases have cleared it (band_dense +0.0041 and
# base224 +0.0070, plus a second base224 seed at +0.0060, were found unpriced on disk after this arm was
# written), but none of them was a
# distillation-plus-blended-labels corner, and this one is.
#
# deliberate: --slot-dropout 0.40 (the legislated operating point, knee-yotj8)
# deliberate: --epochs 24 (the budget half of the pair; on both non-labelwise bases the dropout-only
#             12-epoch variant read BELOW its long twin, so the doubling carries weight)
# reading: stack_increments_drop40_long_s2026.holdout.npz
# bead: knee-p29bb
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
# control  stack_increments_s2026 = 0.8931 ledger75 @e9 (declared 2026-09-13 from the ledger75 pool read)
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

BLEND="$PROCESSED/labels/blend_three_v1.csv"
require "$LABELS"
require "$BLEND"
require "$PROCESSED/raptor_teacher.npz"

run_train stack_increments_drop40_long \
  $RECIPE --epochs 24 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --distill-weight 0.3 --teacher-targets "$PROCESSED/raptor_teacher.npz" \
  --slot-dropout 0.40 \
  --labels "$BLEND" --eval-labels "$LABELS" --out "$MODELS/stack_increments_drop40_long_s2026.pt"
