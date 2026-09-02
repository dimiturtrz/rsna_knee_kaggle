#!/usr/bin/env bash
# Does the encoder learning rate carry the gain, or did depth?
#
# An earlier arm raised the rate 8e-6 -> 4e-5 AND opened 6 -> 12 blocks, and scored +0.0044. Two
# variables in one arm, so the attribution was a guess. This holds depth at 6 and moves only the rate.
#
# control  rank24_s2026            0.8272
# result   2026-08-11              0.8323  (+0.0051, rate alone)
# and the 12-block version of the same rate scored 0.8316 - WORSE by 0.0007. Depth is not a lever.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train rate_isolate \
  $RECIPE --epochs 24 \
  --backbone-lr 4e-5 --labels "$LABELS" --out "$MODELS/lr5x_d6_s2026.pt"
