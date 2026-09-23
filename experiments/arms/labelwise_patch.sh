#!/usr/bin/env bash
# bead: knee-hbta3
# reading: labelwise_patch_s2026.holdout.npz
#
# WHAT THIS ARM IS
#   `labelwise_depth24_drop40_long`'s byte-identical line with ONE variable moved:
#   `--slice-pooling labelwise -> labelwise_patch`. Every diagnosis keeps its softmax over SLICES and
#   gains a second, gated attention-MIL over all slice x patch tokens of the slot, entering through a
#   zero-initialised projection -- so at step zero this model IS the control (unit-tested end to end).
#
# WHY. Every in-plane selection tried so far was label-AGNOSTIC: `focal` and `sided` change the one token
#   collapse all twelve findings share, and both closed flat (docs/AXES.md:109, :240); the ROI loose box
#   and roinative read flat (AXES:84). This is label-SPECIFIC in-plane selection: a meniscal tear and an
#   effusion may read different regions of the same slice.
#
# PRE-REGISTRATION (delta = this arm minus labelwise_depth24_drop40_long_s2026 0.9053 ledger75 @e18)
#   >= +0.0150   clears the bar: per-diagnosis patch selection is a lever; re-read on the other bases.
#   +0.0062..+0.0149
#                above the paired floor, below the bar -- directional, keep for the pool, no slot.
#   <= +0.0061   closes label-specific in-plane attention at this corner: a module that could stay
#                labelwise and was free to select patches did not buy anything.
#   peak         if it peaks ON ITS LAST EPOCH the budget binds and no verdict may be read.
#
# CARD. `--vram-gb 28` inherited. The patch path adds the per-slot patch tensor and its gate activations;
#   measured in the arm_check smoke (knee-hbta3). Nothing else of ours may share the card (`card_check`).
#
# deliberate: --slice-pooling labelwise_patch — the single variable (knee-hbta3).
# deliberate: every other flag inherited verbatim from labelwise_depth24_drop40_long.
#
# control  labelwise_depth24_drop40_long_s2026 (on disk, 0.9053 ledger75, same recipe line). No control run is owed.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train labelwise_patch \
  $RECIPE --epochs 24 --slice-pooling labelwise_patch --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 24 --batch 10 --vram-gb 28 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --hold-annotations --slot-dropout 0.40 \
  --labels "$LABELS" --out "$MODELS/labelwise_patch_s2026.pt"
