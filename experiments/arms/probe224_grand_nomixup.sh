#!/usr/bin/env bash
# bead: knee-feb62.11
# reading: probe224_grand_nomixup_s2026.holdout.npz
#
# Leave-one-group-out over probe224_grand (0.8776@l75, failed its gate against control 0.8914): grand minus mixup.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
run_train probe224_grand_nomixup   --config experiments/recipes/base.toml --config experiments/recipes/grand.toml --config experiments/recipes/probe224.toml   --set regularisation.mixup=0.0   --labels "$LABELS" --out "$MODELS/probe224_grand_nomixup_s2026.pt"
