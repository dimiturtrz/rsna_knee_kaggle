#!/usr/bin/env bash
# bead: knee-0e2gd
# reading: erase_own_cells_s2026.holdout.npz
#
# Blank one cell of each study's own 4x4 occlusion map during training, drawn in proportion to what the
# member leans on there, so the head cannot ride a single region. The map need not be anatomically
# correct: it marks what THIS model uses by construction, which is all the augmentation consumes -- which
# is why it survives knee-j19z1 (the locator gate the same maps failed, twice).
# Maps: labelwise_dense_drop40_long_s2026 swept over 3498 train studies at grid 4.
# Rate 0.25: over 12 epochs a study sees its peak cell present in 9 passes and blanked in 3 -- enough to
# force redundant evidence, few enough that the unerased distribution the holdout serves stays the
# majority.
# ALONE against probe224_control_s2026 0.8914@l75, never inside a composition: a fragment's sign does not
# transfer out of one. Directional only at this price.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
require "$MODELS/patches/labelwise_dense_drop40_long_s2026_g4_train.named.patches.npz"
run_train erase_own_cells \
  --config experiments/recipes/base.toml --config experiments/recipes/trunks/probe224.toml \
  --patch-maps "$MODELS/patches/labelwise_dense_drop40_long_s2026_g4_train.named.patches.npz" \
  --patch-erase 0.25 \
  --labels "$LABELS" --out "$MODELS/erase_own_cells_s2026.pt"
