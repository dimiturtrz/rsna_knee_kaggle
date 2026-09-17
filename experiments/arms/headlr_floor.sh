#!/usr/bin/env bash
# headlr_floor.sh — the shipping line with the head at 5e-5, the third point on the head_lr curve.
#
# bead: knee-h25fx
# reading: headlr_floor_s2026.holdout.npz
# control  labelwise_depth24_drop40_long_s2026 = 0.8679 (midpoint; 0.9053 ledger75, peak e18 of 24)
#
# THE FLAG LINE IS labelwise_depth24_drop40_long's, BYTE FOR BYTE, PLUS ONE FLAG — the same line
# `headlr_ship` carries, differing from it only in the value of --head-lr. --batch 10 and --vram-gb 28
# are part of that line and may not move: RECIPE carries --rank-weight 1.0, whose pairwise population IS
# the batch (head_trial.py:231), so a batch trim makes this measure two knobs and owes a matched control.
#
# TWO POINTS ARE A SIGN, THREE ARE A SHAPE. head_lr is 0.001 in 286 of 288 runs in this corpus, so the
# shipping value is a silent default rather than a tuned one. With 0.001 and 2e-4 alone, a win at 2e-4
# cannot be told apart from a flat top the corner already sits on — the misreading knee-dry3h caught on
# weight_decay, where +0.0106 decomposed into +0.0077 and +0.0029 and the headline was mostly "0.005 is
# too little decay". 5e-5 is what makes the curve readable either way.
#
# IF THIS OOMs, LOWER --slices, NEVER --batch. The batch is what keeps the comparison single-knob.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train headlr_floor \
  $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 24 --batch 10 --vram-gb 28 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --hold-annotations --slot-dropout 0.40 \
  --head-lr 5e-5 \
  --labels "$LABELS" --out "$MODELS/headlr_floor_s2026.pt"
