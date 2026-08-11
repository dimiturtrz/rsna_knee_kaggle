#!/usr/bin/env bash
# Do the two surviving levers ADD?
#
# Rate governs how far the encoder moves off its natural-image features; max pooling governs what a
# three-slice group shows it. Different mechanisms, so they should compose - and if they do not, one
# of them was never doing what its isolated number suggested.
#
# control  rank24_s2026            0.8272
# rate alone (rate_isolate)        0.8323   +0.0051
# pooling alone (slice_count Q)    0.8310   +0.0038
# ADDITIVE PREDICTION              0.8361   - written before the run, so 'as expected' cannot be
#                                             decided after seeing the number.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train compose_small \
  --fold 0 --epochs 24 --one-cycle --weight-floor --rank-weight 1.0 \
  --backbone-lr 4e-5 --slice-pooling max --labels "$LABELS" --out "$PROCESSED/compose_small_s2026.pt"
