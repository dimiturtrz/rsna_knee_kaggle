#!/usr/bin/env bash
# bead: knee-vgz5b.10
# reading: roinative_labelwise_drop40_long_s2026.holdout.npz
#
# roi_labelwise_drop40_long with ONE change: the three sagittal ROI views are re-cut from DICOM decoded
# at 672 px (0.193 mm/px) instead of upsampled from the 336 px cache. Same boxes, same slots, same recipe.
# The roi arm's upsampled crops lost Fracture -0.023 / MCL -0.024 vs control; 73% of sagittal series are
# acquired finer than 0.387 mm/px, so the upsample threw away detail the crop existed to buy.
#
# control  roi_labelwise_drop40_long_s2026 (ledger75 0.8984 on 909 rows).
# OFFLINE ONLY: locator fitted on SKM-TEA, licence ruling knee-vgz5b.7.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train roinative_labelwise_drop40_long \
  $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --scheme roinative \
  --calibrated-targets --hold-annotations \
  --stochastic-depth 0.05 --batch 4 --accumulate 3 --slot-dropout 0.40 \
  --labels "$LABELS" --out "$MODELS/roinative_labelwise_drop40_long_s2026.pt"
