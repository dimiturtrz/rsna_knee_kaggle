#!/usr/bin/env bash
# bead: knee-rn9py
# reading: ctl_nofloor_s2026.holdout.npz
#
# Control-recipe leave-one-out at 224px: weight_floor off, so an unmentioned finding stops pulling at 0.25.
# Read against probe224_control_s2026 0.8914@l75 (909 rows, epoch 10, converged), the same recipe with
# this one fragment left in. INSTRUMENT: a 12-epoch delta ranks the fragment, it does not prove it.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
run_train ctl_nofloor   --config experiments/recipes/base.toml --config experiments/recipes/probe224.toml   --set targets.weight_floor=false   --labels "$LABELS" --out "$MODELS/ctl_nofloor_s2026.pt"
