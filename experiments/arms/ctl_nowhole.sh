#!/usr/bin/env bash
# bead: knee-rn9py
# reading: ctl_nowhole_s2026.holdout.npz
#
# Control-recipe leave-one-out at 224px: whole_stack off.
# Read against probe224_control_s2026 0.8914@l75 (909 rows, epoch 10, converged), the same recipe with
# this one fragment left in. INSTRUMENT: a 12-epoch delta ranks the fragment, it does not prove it.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
run_train ctl_nowhole   --config experiments/recipes/base.toml --config experiments/recipes/probe224.toml   --set encoder.whole_stack=false   --labels "$LABELS" --out "$MODELS/ctl_nowhole_s2026.pt"
