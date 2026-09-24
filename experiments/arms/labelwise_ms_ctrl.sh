#!/usr/bin/env bash
# bead: knee-5jfix
# reading: labelwise_ms_ctrl_s2026.holdout.npz
#
# Control for labelwise_ms: the shipping labelwise recipe on the canonical 336px/12-slice cache
# (0.2-0.8 band, image centre), the only geometry the multiscale cache shares. No labelwise run on
# this cache exists — labelwise224 is 224px — so the multiscale arm has nothing to be read against
# without it.
#
# deliberate: --slices 12 --band-low 0.2 --band-high 0.8 — the multiscale cache's own slab.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train labelwise_ms_ctrl \
  $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --batch 10 --vram-gb 22 --band-low 0.2 --band-high 0.8 \
  --calibrated-targets --hold-annotations --slot-dropout 0.40 \
  --labels "$LABELS" --out "$MODELS/labelwise_ms_ctrl_s2026.pt"
