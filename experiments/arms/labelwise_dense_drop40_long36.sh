#!/usr/bin/env bash
# bead: knee-pqfbs
# reading: labelwise_dense_drop40_long36_s2026.holdout.npz
#
# labelwise_dense_drop40_long at 36 epochs instead of 24: the 24-epoch run peaked e23/24, budget-bound
# (ledger75 2026-09-15). Also the HORIZON-ONLY control for grand_recipe: grand = this line + fragments.
# deliberate: --epochs 36 — the grand recipe horizon, so the pair differs in fragments alone.

source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train labelwise_dense_drop40_long36 \
  $RECIPE --epochs 36 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --slot-dropout 0.40 \
  --labels "$LABELS" --out "$MODELS/labelwise_dense_drop40_long36_s2026.pt"
