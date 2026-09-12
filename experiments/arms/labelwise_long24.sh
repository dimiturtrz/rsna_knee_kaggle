#!/usr/bin/env bash
# labelwise_long24.sh -- the budget question re-asked at the corner where it was never answered.
#
# bead: knee-pfqtw
# reading: labelwise_long24_s2026.holdout.npz
#
# WHY THIS IS NOT dropstack_long AGAIN. `dropstack_long` asked whether 24 epochs beat 12 at the
# `max`-pooling corner and answered NO in its own bottom band (0.8641 vs 0.8654, peak e13 then eleven
# declining epochs). That closes budget AT THAT CORNER. `labelwise` pooling is a different corner: it
# is the family that owns five of the top twelve on the 0.75 ruler while ranking 15-40 at 0.5, and no
# labelwise arm has ever run past twelve epochs. One variable against `labelwise_drop`: --epochs 24.
#
# THIS ARM ONLY RUNS IF `labelwise_drop` CLEARED 0.8663. That is the watcher's gate, not a suggestion:
# if the regulariser did not transfer to labelwise pooling, stretching its budget tests nothing.
#
# PRE-REGISTERED BANDS, against labelwise_drop's own landed number (the watcher prints it).
#   >= +0.0100  budget IS a lever at the labelwise corner and was not at the max corner. That is a
#               pooling x budget interaction, and it re-opens an axis the ledger closed yesterday.
#   +0.0050 to +0.0099  directional only. Seat for diversity; do not re-open the axis on it.
#   -0.0049 to +0.0049  budget is closed at BOTH corners. Say so and stop asking.
#   <= -0.0050  the stretched one-cycle hurts here as it did there, and the mechanism is the schedule
#               rather than the pooling.
#
# READ PEAK-TO-PEAK ONLY. $RECIPE carries --one-cycle, so the anneal stretches with the budget and the
# two runs are NOT comparable epoch-by-epoch -- they differ from epoch 1. This cost a near-miss on
# dropstack_long and is written here so it cannot be repeated.
#
# PRICE: ~1.9 h (280 s/epoch x 24), no submission slot.
# EXPECTED GAIN: P(clears +0.010) ~ 0.15. The prior is bad on purpose -- the same operation at the
#        neighbouring corner just landed negative. This is an INSTRUMENT: it closes the budget axis at
#        both corners instead of one, and the contraction finding says a new member is near-worthless
#        to the board anyway, so the card's best use is retiring a question rather than adding a seat.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train labelwise_long24 \
  $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --stochastic-depth 0.05 \
  --labels "$LABELS" --out "$MODELS/labelwise_long24_s2026.pt"
