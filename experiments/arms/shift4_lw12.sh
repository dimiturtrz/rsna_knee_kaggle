#!/usr/bin/env bash
# bead: knee-ypu31
# reading: shift4_lw12_s2026.holdout.npz
#
# labelwise_ms_ctrl plus --slice-shift 4: the previous and next slice's tokens added, per channel,
# before each of the last four blocks through zero-initialised gates, so epoch 0 starts at the
# control's function. Read against labelwise_ms_ctrl (0.9006 ledger75), the same cache, recipe and seed.
#
# deliberate: --slices 12 --band-low 0.2 --band-high 0.8 — the control's slab, the cheap line.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train shift4_lw12 \
  $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --batch 10 --vram-gb 22 --band-low 0.2 --band-high 0.8 \
  --calibrated-targets --hold-annotations --slot-dropout 0.40 --slice-shift 4 \
  --labels "$LABELS" --out "$MODELS/shift4_lw12_s2026.pt"
