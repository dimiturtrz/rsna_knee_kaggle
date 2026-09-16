#!/usr/bin/env bash
# bead: knee-3tk4b
# reading: probe224_grand_trim_s2026.holdout.npz
#
# Every fragment the knee-feb62.11 leave-one-out priced POSITIVE, dropped together, on the same
# probe224 recipe as the five runs it is read against: grand 0.8776, control 0.8914, nomixup 0.8806
# (+0.0030), nohard 0.8789 (+0.0013), noaug 0.8726 (-0.0050, so the aug block STAYS).
# It answers additivity before a 24-epoch seat is spent on the assumption.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
run_train probe224_grand_trim \
  --config experiments/recipes/base.toml --config experiments/recipes/grand.toml --config experiments/recipes/probe224.toml \
  --set regularisation.mixup=0.0 --set sampling.hard_start=1.0 --set sampling.hard_ramp=0.0 \
  --labels "$LABELS" --out "$MODELS/probe224_grand_trim_s2026.pt"
