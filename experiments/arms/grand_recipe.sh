#!/usr/bin/env bash
# bead: knee-feb62.4
# reading: grand_recipe_s2026.holdout.npz
#
# The grand recipe (knee-feb62): every composable fragment on one progress curve, 36 epochs. The
# recipe lives in experiments/recipes/grand.toml; flags here only name the labels and the checkpoint.

source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train grand_recipe \
  --config experiments/recipes/grand.toml \
  --labels "$LABELS" --out "$MODELS/grand_recipe_s2026.pt"
