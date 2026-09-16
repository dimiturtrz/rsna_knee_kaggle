#!/usr/bin/env bash
# bead: knee-rn9py
# reading: ctl_fastbb_s2026.holdout.npz
#
# Control-recipe leave-one-out at 224px: backbone_lr 4e-5 -> 1e-4; a DIRECTION, not a removal.
# Read against probe224_control_s2026 0.8914@l75 (909 rows, epoch 10, converged), the same recipe with
# this one fragment left in. INSTRUMENT: a 12-epoch delta ranks the fragment, it does not prove it.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
run_train ctl_fastbb   --config experiments/recipes/base.toml --config experiments/recipes/probe224.toml   --set encoder.backbone_lr=1e-4   --labels "$LABELS" --out "$MODELS/ctl_fastbb_s2026.pt"
