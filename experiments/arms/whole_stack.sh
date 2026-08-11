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
# cost     MEASURED, not estimated. A step puts 12 studies x 6 slots x 12 slices = 864 images through
#          the encoder. The first attempt at batch 12 without checkpointing pinned all 32 GB of the
#          5090 and thrashed - Windows pages GPU memory over PCIe instead of failing, so utilisation
#          read 100% while power sat at 145 W of 575 and fifteen minutes produced no epoch at all.
#          --recompute-activations is therefore not optional here.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train whole_stack \
  --fold 0 --epochs 24 --one-cycle --weight-floor --rank-weight 1.0 \
  --slice-pooling max --whole-stack --recompute-activations --labels "$LABELS" --out "$PROCESSED/wholestack_s2026.pt"
