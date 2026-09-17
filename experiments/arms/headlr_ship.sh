#!/usr/bin/env bash
# headlr_ship.sh — the shipping line with the head slowed from 25x the trunk to 5x.
#
# bead: knee-xuarh
# reading: headlr_ship_s2026.holdout.npz
# control  labelwise_depth24_drop40_long_s2026 = 0.8679 (midpoint; 0.9053 ledger75, peak e18 of 24)
#
# THE FLAG LINE IS labelwise_depth24_drop40_long's, BYTE FOR BYTE, PLUS ONE FLAG. --batch 10 and
# --vram-gb 28 are part of that line and may not move: RECIPE carries --rank-weight 1.0, whose pairwise
# population IS the batch (head_trial.py:231), so a batch trim makes this measure two knobs and owes a
# matched control of its own.
#
# head_lr IS A SILENT DEFAULT ON THE SHIPPING LINE — the arm names it nowhere, so it resolves to 0.001
# against a backbone pinned at 4e-5. The corpus holds exactly one isolating pair on this key, at the
# 224/max corner: 0.005 -> 0.0002 reads +0.0106 midpoint. Shipping sits INSIDE that interval, so the
# local sign is assumed, not measured. 2e-4 is the step that keeps the arm on the measured side of it.
#
# THE HEAD HERE IS NOT THE HEAD THE PAIR WAS MEASURED ON. That pair ran --slice-pooling max, one shared
# projection; this line runs labelwise, twelve per-label projections chasing twelve columns. More head
# capacity at 25x the trunk rate is the mechanism, and it is stronger here than where the pair was taken.
#
# IF THIS OOMs, LOWER --slices, NEVER --batch. The batch is what keeps the comparison single-knob.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train headlr_ship \
  $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 24 --batch 10 --vram-gb 28 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --hold-annotations --slot-dropout 0.40 \
  --head-lr 2e-4 \
  --labels "$LABELS" --out "$MODELS/headlr_ship_s2026.pt"
