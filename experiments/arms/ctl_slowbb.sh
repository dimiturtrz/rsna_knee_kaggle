#!/usr/bin/env bash
# bead: knee-lm2ol
# reading: ctl_slowbb_s2026.holdout.npz
#
# The other side of the one continuous knob knee-rn9py's sweep gave a sign to. ctl_fastbb put backbone_lr
# at 1e-4 (2.5x the control) and read -0.0215; this puts it at 2e-5 (0.5x). Same cache, same 909-row ruler,
# 12 epochs, against probe224_control_s2026 0.8914@l75.
# Branches pre-registered in knee-lm2ol: >=0.8964 licenses a full-res arm, 0.8864-0.8964 flat, <0.8864 peak.
# The named harness suspect is the 12-epoch one_cycle budget -- a flat AND undertrained result defers the
# branch to an epoch-budget arm, it does not close the axis.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
run_train ctl_slowbb \
  --config experiments/recipes/base.toml --config experiments/recipes/probe224.toml \
  --set encoder.backbone_lr=2e-5 \
  --labels "$LABELS" --out "$MODELS/ctl_slowbb_s2026.pt"
