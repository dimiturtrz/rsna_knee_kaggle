#!/usr/bin/env bash
# bead: knee-feb62.11
# reading: probe224_grand_noaug_s2026.holdout.npz
#
# Leave-one-group-out over probe224_grand (0.8776@l75, failed its gate against control 0.8914): grand minus aug.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
run_train probe224_grand_noaug   --config experiments/recipes/base.toml --config experiments/recipes/grand.toml --config experiments/recipes/probe224.toml   --set regularisation.ghosting=0.0 --set regularisation.intensity_aug=false --set regularisation.scale_jitter=0.08 --set regularisation.aug_start=1.0 --set regularisation.aug_end=1.0 --set regularisation.aug_ramp=0.0 --set regularisation.aug_tail=0.0   --labels "$LABELS" --out "$MODELS/probe224_grand_noaug_s2026.pt"
