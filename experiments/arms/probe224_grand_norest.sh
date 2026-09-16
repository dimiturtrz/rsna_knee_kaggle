#!/usr/bin/env bash
# bead: knee-dcl6e
# reading: probe224_grand_norest_s2026.holdout.npz
#
# The four fragments the knee-feb62.11 leave-one-out never varied, returned to their control values in
# one group. Named by diffing the two arms' own recipe stamps, which is the only place three of them
# appear: the control sets none of these keys, it inherits the defaults.
# Accounted for so far, against grand 0.8776 and control 0.8914: mixup +0.0030, hard +0.0013,
# ema +0.0032, aug -0.0050 -- positives +0.0075 of a 0.0138 deficit, leaving -0.0063 unexplained.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
run_train probe224_grand_norest \
  --config experiments/recipes/base.toml --config experiments/recipes/grand.toml --config experiments/recipes/probe224.toml \
  --set regularisation.stochastic_depth=0.0 --set regularisation.padding_mode=border \
  --set optim.warmup=0.15 --set optim.patience_from_progress=0.0 \
  --labels "$LABELS" --out "$MODELS/probe224_grand_norest_s2026.pt"
