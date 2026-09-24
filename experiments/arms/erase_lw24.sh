#!/usr/bin/env bash
# bead: knee-xxr8l
# reading: erase_lw24_s2026.holdout.npz
#
# `labelwise_depth24_drop40_long`'s line with ONE addition: erase_own_cells' map-driven erasing
# (--patch-maps <g4 train> --patch-erase 0.25). knee-0e2gd read it null ALONE on the 224/12-epoch probe
# and left this use open; the maps were swept from this very family, so here the map is the one the
# shipping recipe leans on.
#
# PRE-REGISTRATION (delta = this arm minus labelwise_depth24_drop40_long_s2026 0.9053 ledger75 @e18)
#   >= +0.0150   lever: full-fit refit and a slot.
#   +0.0062..+0.0149  directional, pool only.
#   <= +0.0061   erasing closed on both families.
#   peak on its last epoch: budget binds, no verdict.
#
# deliberate: --patch-maps/--patch-erase 0.25 — the single addition, rate inherited from erase_own_cells.
# control  labelwise_depth24_drop40_long_s2026 (on disk, 0.9053 ledger75). No control run is owed.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"
require "$MODELS/patches/labelwise_dense_drop40_long_s2026_g4_train.named.patches.npz"

run_train erase_lw24 \
  $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 24 --batch 10 --vram-gb 22 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --hold-annotations --slot-dropout 0.40 \
  --patch-maps "$MODELS/patches/labelwise_dense_drop40_long_s2026_g4_train.named.patches.npz" \
  --patch-erase 0.25 \
  --labels "$LABELS" --out "$MODELS/erase_lw24_s2026.pt"
