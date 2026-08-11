#!/usr/bin/env bash
# Read the whole cached stack per step instead of one three-slice window.
#
# A training step currently sees 6 slots x 3 consecutive slices = 18 images, out of 72 cached and
# ~165-190 acquired. A finding outside the sampled window is invisible to that step. Whole-stack
# pooling removes the sampled-group limitation without changing what a group SPANS - which is the
# confound that killed the slice-count arm - and it makes training match a deployment that already
# averages ten overlapping windows.
#
# Two measurements say the stack holds more than one window shows: overlap TTA is worth +0.0156 at
# inference, and the occlusion depth profile ranges 0.7976 to 0.8332 across windows.
#
# control  compose_small / arm Q (max pooling, WINDOWED). Single variable: the read.
# cost     ~4x a step - 12 encoder passes per slot against 3.
source "$(dirname "$0")/../lib.sh"
require "$LABELS"

run_train whole_stack \
  --fold 0 --epochs 24 --one-cycle --weight-floor --rank-weight 1.0 \
  --slice-pooling max --whole-stack --labels "$LABELS" --out "$PROCESSED/wholestack_s2026.pt"
echo "ARM DONE whole_stack"
