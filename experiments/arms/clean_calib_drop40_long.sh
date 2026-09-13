#!/usr/bin/env bash
# clean_calib_drop40_long.sh — knee-rpugl. `clean_calib.sh` with the operator pair and NOTHING ELSE.
#
# THE FINDING THIS ARM IS BUILT ON, measured 2026-09-13 on the full 909, LogitAverage, ledger75 cuts,
# against the honest core six at 0.9050. The pair (`--slot-dropout 0.40` + `--epochs 12 -> 24`) is a
# SEAT-REPLACEMENT operator, not a seat-addition one, and the two readings disagree sharply:
#
#   ADDED as a seventh seat        SWAPPED for the base it upgrades
#   chainfour_up   +0.0007         chainfour        +0.0012
#   blendhold_up   +0.0000         blendhold        +0.0008
#   dense_up       +0.0020         labelwise_dense  +0.0014
#   ext_up         +0.0017         labelwise_ext    +0.0008
#                                  all four         +0.0032
#                                  all four + depth_up (n=7)  0.9093  +0.0043
#
# chainfour_up and blendhold_up each clear the operator band SOLO (+0.0070, +0.0050 over their bases) and
# then add ~nothing as extra seats — the signature of a correlated duplicate. Remove the base it
# duplicates and the same member is worth +0.0012. Note what that costs the entry currently in flight:
# the four-swap core plus depth_up reads 0.9093 at n=7, BEATING tripleseat's nine at 0.9090 with two
# fewer seats and two fewer members to mount.
#
# NON-LABELWISE CARRIES AS MUCH AS LABELWISE ON THIS AXIS — chainfour is the single best swap of the four.
# That is what licenses spending a card on the distillation corner, which the ADD reading would have
# closed.
#
# NO SINGLE SWAP CLEARS THE BOARD RESOLUTION AND THAT IS NOT THE CLAIM. The composite is: six swaps plus
# depth_up projects to ~+0.0058 against a +0.0040 seat floor. `clean_calib_s2026` (0.8940) and
# `warm_calib_lowlr_s2026` (0.8910) are the only two core seats with no upgraded twin on disk; this arm
# builds the first of the two.
#
# ONE CHANGE, TWO FLAGS. Every flag below is `clean_calib.sh`'s, byte for byte, plus `--slot-dropout 0.40`
# and `--epochs 12 -> 24`. 0.40 is the operating point legislated in knee-yotj8, not a swept knob, and the
# budget doubling is the same pair that ran on five bases before this one. Expect exactly
# `slot_dropout 0.0->0.4`, `epochs 12->24` and `out` in the recorded-recipe diff.
#
# PRE-REGISTERED, before the run. Swap value over the FOUR-swap core at 0.9082, replacing
# `clean_calib_s2026` with this checkpoint:
#   >= +0.0008   seats the swap; `warm_calib_lowlr` takes the next card and the six-swap entry is built.
#   +0.0001..+0.0007  directional. Keep the member, no further arm on this axis.
#   <= 0.0000    the operator does not reach the distillation corner, and the six-swap projection is void.
#   solo floor   >= 0.8940 (its own base), or the transfer failed HERE and the swap read is void.
#   peak         if this peaks on its LAST epoch, read `convergence_check` (knee-a21h9) — argmax-last AND
#                a final-three-epoch gain over the 0.0062 holdout floor is what refuses a verdict, not
#                argmax-last alone. `labelwise_dense_drop40_long` peaked at e23 of 0..23 and its verdict
#                stood on a tail gain of +0.0002.
#
# PRICE: ~2.4 h (24 epochs at the parent's measured 363 s/epoch). P(clears +0.0008) ~ 0.55.
#
# deliberate: --slot-dropout 0.40 (the legislated operating point, knee-yotj8)
# deliberate: --epochs 24 (the budget half of the pair; the dropout-only 12-epoch variants read BELOW
#             their long twins on both non-labelwise bases, so the doubling carries weight)
# reading: clean_calib_drop40_long_s2026.holdout.npz
# bead: knee-rpugl
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
# control  clean_calib_s2026 = 0.8940 ledger75 (declared 2026-09-13 from the ledger75 pool read)
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

TEACHER="$PROCESSED/teacher_clean.npz"
require "$TEACHER"

run_train clean_calib_drop40_long \
  $RECIPE --epochs 24 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --distill-weight 0.7 --teacher-targets "$TEACHER" \
  --calibrated-targets \
  --slot-dropout 0.40 \
  --labels "$LABELS" --out "$MODELS/clean_calib_drop40_long_s2026.pt"
