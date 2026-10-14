#!/usr/bin/env bash
# bead: knee-feb62.11
# reading: probe224_grand_noema_s2026.holdout.npz
#
# Leave-one-group-out over probe224_grand (0.8776@l75, failed its gate against control 0.8914): grand minus ema.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
run_train probe224_grand_noema   --config experiments/recipes/base.toml --config experiments/recipes/grand.toml --config experiments/recipes/trunks/probe224.toml   --set optim.ema=false   --labels "$LABELS" --out "$MODELS/probe224_grand_noema_s2026.pt"
