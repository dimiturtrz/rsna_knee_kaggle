#!/usr/bin/env bash
# bead: knee-eduxz
# reading: probe224_control_noema_s2026.holdout.npz
#
# probe224_control with EMA off, one change, read against probe224_control 0.8914@l75 already on disk.
# The knee-feb62.11 noema probe moved +0.0032 inside the grand composition, and EMA is in the control
# and in every champion-family seat too -- so that move is either a cross-recipe lever or an interaction
# with the grand fragments, and only the base recipe can tell the two apart.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
run_train probe224_control_noema \
  --config experiments/recipes/base.toml --config experiments/recipes/probe224.toml \
  --set optim.ema=false \
  --labels "$LABELS" --out "$MODELS/probe224_control_noema_s2026.pt"
