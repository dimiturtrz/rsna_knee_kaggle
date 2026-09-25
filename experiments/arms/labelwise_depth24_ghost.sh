#!/usr/bin/env bash
# labelwise_depth24_ghost.sh — the shipping line plus KMAR-calibrated ghosting, one variable.
#
# bead: knee-mum8w
# reading: labelwise_depth24_ghost_s2026.holdout.npz
#
# `ghosting_s2026` read +0.0057 directional on the cross-site decode axis, on the max-pool 20-slice line;
# the operator never ran on the labelwise 24-slice line that ships. holdout909 cannot see a robustness
# lever, so the holdout is only the safety gate (>= 0.8990 ledger75) and the reading is a board seat swap
# against `tripleseat_v2`. Severity 0.040 is KMAR-50K's measured median motion residual (see ghosting.sh).
# `--vram-gb 22` is the standing cap; the control peaked at 15.9 GB allocated.

source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train labelwise_depth24_ghost \
  $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 24 --batch 10 --vram-gb 22 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --hold-annotations --slot-dropout 0.40 \
  --ghosting 0.040 \
  --labels "$LABELS" --out "$MODELS/labelwise_depth24_ghost_s2026.pt"
