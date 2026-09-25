#!/usr/bin/env bash
# bead: knee-6u1tq
# reading: shift4_depth24_s2026.holdout.npz
#
# labelwise_depth24_drop40_long (0.9053 ledger75, the shipping solo) plus --slice-shift 4, which read
# +0.0035 on the 12-slice labelwise line with its gates fired (knee-ypu31). One flag moves, and the
# VRAM ask drops from the base's 28 to the owner's cap of 22.
#
# deliberate: --slices 24 --batch 10 --slot-dropout 0.40 --epochs 24 — inherited verbatim from the base.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train shift4_depth24 \
  $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 24 --batch 10 --vram-gb 22 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --hold-annotations --slot-dropout 0.40 --slice-shift 4 \
  --labels "$LABELS" --out "$MODELS/shift4_depth24_s2026.pt"
