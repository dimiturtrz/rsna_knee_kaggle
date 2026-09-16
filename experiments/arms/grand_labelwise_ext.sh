#!/usr/bin/env bash
# bead: knee-feb62.7
# reading: grand_labelwise_ext_s2026.holdout.npz
# DRAFT, not armed. Seat labelwise_ext_drop40_long_s2026 under grand.toml; seat identity = extended scheme.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train grand_labelwise_ext \
  --config experiments/recipes/base.toml --config experiments/recipes/grand.toml \
  --scheme extended --batch 4 --accumulate 3 \
  --labels "$LABELS" --out "$MODELS/grand_labelwise_ext_s2026.pt"
