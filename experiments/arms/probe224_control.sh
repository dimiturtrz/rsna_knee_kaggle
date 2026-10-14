#!/usr/bin/env bash
# bead: knee-feb62.8
# reading: probe224_control_s2026.holdout.npz
#
# Cheap probe pair for the grand recipe (knee-feb62.4): grand.toml's seat without its fragments, 224 px,
# 12 slices, 12 epochs. Gate and price pre-registered in knee-feb62.8. Recipe in experiments/recipes/trunks/probe224.toml.

source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train probe224_control \
  --config experiments/recipes/base.toml --config experiments/recipes/trunks/probe224.toml \
  --labels "$LABELS" --out "$MODELS/probe224_control_s2026.pt"
