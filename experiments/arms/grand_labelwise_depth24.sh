#!/usr/bin/env bash
# bead: knee-feb62.7
# reading: grand_labelwise_depth24_s2026.holdout.npz
# DRAFT, not armed. Seat labelwise_depth24_drop40_long_s2026 under grand.toml; seat identity = 24 slices, batch 10.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train grand_labelwise_depth24 \
  --config experiments/recipes/base.toml --config experiments/recipes/grand.toml \
  --set data.slices=24 --batch 10 --vram-gb 28 \
  --labels "$LABELS" --out "$MODELS/grand_labelwise_depth24_s2026.pt"
