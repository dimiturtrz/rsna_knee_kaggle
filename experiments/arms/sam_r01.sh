#!/usr/bin/env bash
# sam_r01 — sharpness-aware minimization at rho=0.01 on the best dense recipe.
#
# THE SINGLE VARIABLE IS --sharpness 0.01 AND NOTHING ELSE. Control is calib_dense_hold_s2026
# = 0.8634 LEDGER / 0.8949 LEDGER75 @holdout909, the identical recipe.
#
# rho=0.05 is refuted on this stack (knee-zqouw, experiments/README.md): a uniform ~4-point lag
# from epoch 0, with the ascent witnessed live, which places the cause in the radius rather than
# in the mechanism. 0.01 is the smaller of the two radii that closure named.
#
# DO NOT ARM trajectory_check ON THIS ARM. The hypothesis is that SAM lags early and pays at the
# anneal, so an early-epoch target cannot gate it without presuming the answer. It runs all 12.
#
# The recipe stamps "bf16": "True", so no GradScaler exists on this path and Sharpness never has a
# scale factor to unscale_. --accumulate must stay 1: sharpen() raises rather than approximating a
# perturbation across micro-batches.
#
# NO --vram-gb: DeviceBudget derives the cap from --slices — 20 slices reserve 28.26 GB.
#
# bead: knee-fw1hc
# reading: sam_r01_s2026.holdout.npz
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train sam_r01 \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --sharpness 0.01 \
  --labels "$LABELS" --out "$MODELS/sam_r01_s2026.pt"
