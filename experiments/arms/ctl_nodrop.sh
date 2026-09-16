#!/usr/bin/env bash
# bead: knee-rn9py
# reading: ctl_nodrop_s2026.holdout.npz
#
# Control-recipe leave-one-out at 224px: slot_dropout 0.40 -> 0, the knob the whole drop40_long family is named for.
# Read against probe224_control_s2026 0.8914@l75 (909 rows, epoch 10, converged), the same recipe with
# this one fragment left in. INSTRUMENT: a 12-epoch delta ranks the fragment, it does not prove it.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
run_train ctl_nodrop   --config experiments/recipes/base.toml --config experiments/recipes/probe224.toml   --set regularisation.slot_dropout=0.0   --labels "$LABELS" --out "$MODELS/ctl_nodrop_s2026.pt"
