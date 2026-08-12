#!/usr/bin/env bash
# unfreeze_depth.sh [blocks] — how many of the encoder's twelve blocks should adapt?
#
# THE SPEED IS LARGE AND MEASURED: one epoch at four blocks ran 1075 img/s against six blocks' 817,
# +32%, eager both ways. That stacks with compile's +48% toward roughly twice the experiments per hour,
# which is worth a full arm to settle.
#
# THE SCORE IS NOT SETTLED, and the throughput arm cannot settle it. Its epoch-0 holdout was 0.7539
# against 0.7600 - a 0.0061 gap sitting exactly ON the 0.0062 noise floor, read from the single weakest
# measurement in the ledger. A shallower adaptation could catch up (the head has more epochs to use a
# fixed representation) or fall further behind (fewer blocks to specialise to MRI, which is a long way
# from the natural images the encoder was trained on). Both are plausible; one epoch distinguishes them
# not at all.
#
# WHY IT MIGHT COST NOTHING: our own ledger says capacity is not the constraint here — base over small
# was +0.0011 against the same floor — and unfreezing fewer blocks is a capacity reduction of the same
# kind. WHY IT MIGHT COST SOMETHING: the frozen blocks are the GENERIC ones, and knee MRI is far enough
# from ImageNet that the specialisation may need to reach deeper than four blocks.
#
# control  wsfold0_e6 = 0.8457, the same recipe with six blocks unfrozen at 12 epochs.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
BLOCKS="${1:-4}"
require "$LABELS"

run_train "unfreeze_${BLOCKS}" \
  --fold 0 --epochs 12 --one-cycle --weight-floor --rank-weight 1.0 \
  --slice-pooling max --whole-stack --unfreeze-last "$BLOCKS" \
  --recompute-activations --compile-encoder --vram-gb 26 \
  --labels "$LABELS" --out "$PROCESSED/unfreeze${BLOCKS}_s2026.pt"
