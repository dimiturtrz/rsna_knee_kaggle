#!/usr/bin/env bash
# bead: knee-1a4gm
# reading: probe224_control_aug_s2026.holdout.npz
#
# The knee-feb62.11 decomposition is closed: mixup +0.0030, hard +0.0013, ema +0.0032, the structural
# four +0.0049 (knee-dcl6e), aug -0.0050 -- positives +0.0124 of the 0.0138 deficit, and knee-3tk4b
# measured them composing additively. So the grand recipe is a pile of small harms, and its ONE paying
# fragment is the aug block. This puts that block alone on the CONTROL, the recipe that wins.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
run_train probe224_control_aug \
  --config experiments/recipes/base.toml --config experiments/recipes/trunks/probe224.toml \
  --set regularisation.ghosting=0.04 --set regularisation.intensity_aug=true --set regularisation.scale_jitter=0.25 \
  --set regularisation.aug_start=0.25 --set regularisation.aug_mid=1.0 --set regularisation.aug_end=0.5 \
  --set regularisation.aug_ramp=0.25 --set regularisation.aug_tail=0.2 \
  --labels "$LABELS" --out "$MODELS/probe224_control_aug_s2026.pt"
