#!/usr/bin/env bash
# bead: knee-jmct4
# reading: ctl_stripall_s2026.holdout.npz
#
# The five knee-rn9py fragments removed TOGETHER, not one at a time. The sweep's +0.0253 running total
# is arithmetic over five separate arms; this is the only arm that measures it. Same cache, same 909-row
# ruler, same 12 epochs, against probe224_control_s2026 0.8914@l75.
# Branches are pre-registered in knee-jmct4: 0.8630-0.8690 additive, >0.8750 overlapping, <0.8570 interacting.
# INSTRUMENT -- it strips a recipe and cannot raise a score; its value is that it can falsify a written number.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
run_train ctl_stripall \
  --config experiments/recipes/base.toml --config experiments/recipes/probe224.toml \
  --set regularisation.slot_dropout=0.0 \
  --set model.slice_pooling=max \
  --set targets.calibrated_targets=false \
  --set targets.rank_weight=0.0 \
  --set targets.weight_floor=false \
  --labels "$LABELS" --out "$MODELS/ctl_stripall_s2026.pt"
