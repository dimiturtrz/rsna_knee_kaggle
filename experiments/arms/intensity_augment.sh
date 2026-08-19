#!/usr/bin/env bash
# Gamma, contrast, blur and noise on top of the shipping recipe. Aimed at the scanner, not the model.
#
# THE MECHANISM. MR has no absolute intensity scale. Our reader normalises each series to its own
# 1st-99th percentile, which fixes the LEVEL and the SCALE of a series but not the SHAPE of its
# contrast — two scanners can render the same tissue contrast with different gamma and both survive a
# percentile normalisation unchanged. Intensity augmentation is the standard answer, and it is the one
# axis of augmentation this project has never measured on the current ruler.
#
# WHY THIS AXIS AND NOT ANOTHER. The largest subgroup spread we have measured is by MANUFACTURER:
# 0.8020 to 0.8256 across groups, wider than the report-language gap that knee-crl was deflated over,
# and language turned out to be confounded with site anyway. A model invariant to contrast rendering
# is the direct attack on that axis. The corpus is 16 sites in a dozen languages, so it is not a
# hypothetical population.
#
# THE FLAG ALREADY EXISTS AND HAS NEVER BEEN READ HERE. AugmentConfig carries gamma 0.7-1.5, contrast
# 0.85-1.15, a blur and 0.02 noise, defaulting OFF so that enabling it stays an ablatable decision.
# Three checkpoints on disk have "aug" in their names and none records whether it was used - they
# predate options recording (knee-ehn), and their 0.820-0.822 holdouts are on the ruler we rebuilt. So
# there is no prior number for this, only the appearance of one.
#
# NOISE IS DELIBERATELY LOW. 0.02 against the sibling cardiac project's 0.08, because theirs acts on
# z-scored images and ours on [0, 1]. Inheriting the constant rather than the ARGUMENT is how a
# borrowed default becomes a silent mistake.
#
# control  res392_ws_s2026 = 0.8523, the same recipe at the same resolution with geometry-only
#          augmentation. Single variable: the intensity block.
# risk     augmentation trades training signal for invariance, and this model is supervision-limited -
#          58 of 4,407 studies carry an image-read label. If it costs more than it buys, that is a
#          real answer about THIS corpus rather than about augmentation.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

# NO-OP ON THIS MACHINE, kept only so nobody re-adds it. Torch prints 'expandable_segments not
# supported on this platform' on Windows at startup, so this line never reclaimed anything - any
# arm that fitted while carrying it fitted for another reason. Measured 2026-08-13.
wait_for_cache 448

run_train intensity_392 \
  $RECIPE --epochs 12 \
  --slice-pooling max --whole-stack --intensity-aug \
  --image-size 448 --train-size 392 \
  --backbone-lr 4e-5 --recompute-activations \
  --labels "$LABELS" --out "$PROCESSED/intensity392_ws_s2026.pt"
