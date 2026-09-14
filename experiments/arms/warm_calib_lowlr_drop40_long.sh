#!/usr/bin/env bash
# warm_calib_lowlr_drop40_long.sh — knee-rkegi, the operator pair onto the earliest-peaking SEAT.
#
# `warm_calib_lowlr_s2026` reads 0.8910 ledger75 PEAKING AT EPOCH 4 OF 12 and declining for seven. That
# is the earliest peak in the pool — earlier than grade_ladder (@e6), through_dense (@e8),
# stack_increments (@e9). A budget wall peaks on the LAST epoch; peaking at 4 of 12 and declining for
# seven is a missing regulariser, the diagnosis the pair answers.
#
# AND IT IS A SEAT, not a spare: it sits in the best n=7 seating this project has measured. knee-fix1t
# says the ceiling is the POOL, not the seating, so an upgrade that lands on a seated member attacks it
# directly.
#
# THE DOUBT THAT MAKES THIS A QUESTION. This base is WARM-STARTED from `calib_dense_hold.enc.pt`, so its
# trunk arrives annealed. The early peak may be an exhausted trunk rather than a missing regulariser —
# knee-wbwq's own 'THE RAMP WAS NOT THE CAUSE' branch. Then the budget half buys nothing and the dropout
# half fights a trunk that cannot move.
#
# ONE CHANGE, TWO FLAGS. Every flag below is `warm_calib_lowlr.sh`'s, byte for byte, plus
# `--slot-dropout 0.40` and `--epochs 12 -> 24`. 0.40 is the operating point legislated in knee-yotj8,
# not a swept knob. Expect exactly `slot_dropout 0.0->0.4`, `epochs 12->24` and `out` in the
# recorded-recipe diff.
#
# PRE-REGISTERED, before the run, against its own base at 0.8910 ledger75 @e4:
#   >= +0.0040   the pair transfers to warm-started recipes. Roster it, price it as a SWAP for its own
#                base in the n=7 seating, widen onto the rest of the warm-start family.
#   +0.0015..+0.0039  partial. Keep ONLY if it prices positive as a marginal over the seating —
#                stack_increments read +0.0020 solo and -0.0001 seated, and a solo is not a seat.
#   <= +0.0014   the pair does not reach warm starts, and knee-wbwq's annealed-trunk reading gains its
#                first independent support. Closes the warm-start family to this operator.
#   solo floor   >= 0.8910, or the transfer failed HERE and no branch above may be read.
#   peak         if this peaks on its LAST epoch, read `convergence_check` (knee-a21h9) — argmax-last
#                AND a final-three-epoch gain over the 0.0062 holdout floor is what refuses a verdict.
#
# PRICE: ~1.3 h. P(clears +0.0040) ~ 0.70 — eight of nine bases have cleared it and this one carries the
# strongest diagnosis in the pool, discounted for the warm-start doubt above.
#
# deliberate: --slot-dropout 0.40 (the legislated operating point, knee-yotj8)
# deliberate: --epochs 24 (the budget half of the pair)
# reading: warm_calib_lowlr_drop40_long_s2026.holdout.npz
# bead: knee-rkegi
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
# control  warm_calib_lowlr_s2026 = 0.8910 ledger75 @e4 (declared 2026-09-13 from the ledger75 pool read)
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"
require "$MODELS/encoders/calib_dense_hold.enc.pt"

run_train warm_calib_lowlr_drop40_long \
  $RECIPE --epochs 24 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 1e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --encoder-weights "$MODELS/encoders/calib_dense_hold.enc.pt" \
  --slot-dropout 0.40 --vram-gb 20 \
  --labels "$LABELS" --out "$MODELS/warm_calib_lowlr_drop40_long_s2026.pt"
