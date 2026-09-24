#!/usr/bin/env bash
# bead: knee-rt48x
# reading: auc_ft_lw24_s2026.holdout.npz
#
# LibAUC's published recipe is TWO-STAGE: cross-entropy first, then a short AUC-margin fine-tune from the
# fitted network (Yuan et al. 2021, arXiv 2012.03173). `margin_seat` replaced cross-entropy FROM SCRATCH
# and read -0.0053 (README:374), so the use the paper actually prescribes has never run here. Stage one is
# `labelwise_depth24_drop40_long_s2026`, already on disk; this is stage two only.
#
# PRE-REGISTRATION (delta = this arm minus labelwise_depth24_drop40_long_s2026 0.9053 ledger75 @e18)
#   >= +0.0150   lever: run the CE-continuation control, then a slot.
#   +0.0062..+0.0149  directional; the six extra epochs are a confound, CE-continuation control owed.
#   <= +0.0061   two-stage AUC-margin closed.
#
# deliberate: --auc-margin 1.0 --rank-weight 0 — the objective, spelled as margin_seat spelled it.
# deliberate: --init/--epochs 6/--backbone-lr 1e-5 — stage two of the two-stage recipe.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"
PARENT="$MODELS/labelwise_depth24_drop40_long_s2026.pt"
require "$PARENT"

run_train auc_ft_lw24 \
  $RECIPE --epochs 6 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 1e-5 --slices 24 --batch 10 --vram-gb 22 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --hold-annotations --slot-dropout 0.40 \
  --rank-weight 0 --auc-margin 1.0 \
  --init "$PARENT" --init-scope all \
  --labels "$LABELS" --out "$MODELS/auc_ft_lw24_s2026.pt"
