#!/usr/bin/env bash
# sam_ship — sharpness at rho=0.01 composed onto the SHIPPING recipe.
#
# THE SINGLE VARIABLE IS --sharpness 0.01. Control is labelwise_depth24_drop40_long_s2026 = 0.9053
# LEDGER75 @holdout909, the best honest solo on disk, whose flag list this reproduces exactly.
#
# WHY THIS CORNER AND NOT sam_r01's. sam_r01 measures sharpness against calib_dense_hold_s2026 at
# 0.8949 — about a point below where this project ships. A carry there does not produce a shippable
# member, so the escalation knee-fw1hc names is read here, on the recipe that is the deliverable.
#
# THE COMPOSITION RISK, NAMED BEFORE THE RUN. This recipe already carries --slot-dropout 0.40 and 24
# epochs. Sharpness and slot dropout are both regularisers; a gain measured against an unregularised
# control need not survive beside one. A flat read here with sam_r01 carrying is evidence of
# SATURATED REGULARISATION, not of sharpness failing, and closes composition rather than the axis.
#
# DO NOT ARM trajectory_check ON THIS ARM, for knee-fw1hc's reason: the mechanism predicts an early lag
# paid back at the anneal, so an early-epoch target presumes the answer. It runs all 24.
#
# The recipe stamps "bf16": "True", so no GradScaler exists and Sharpness has no scale factor to
# unscale_. --accumulate must stay 1: sharpen() raises rather than approximating across micro-batches.
#
# deliberate: --vram-gb 28 matches the control's value, not the 22 the owner's cap would set. The budget
#             feeds batch sizing, so changing it moves a second variable.
#
# bead: knee-k3v5t
# reading: sam_ship_s2026.holdout.npz
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train sam_ship \
  $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 24 --batch 10 --vram-gb 28 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --hold-annotations --slot-dropout 0.40 \
  --sharpness 0.01 \
  --labels "$LABELS" --out "$MODELS/sam_ship_s2026.pt"
