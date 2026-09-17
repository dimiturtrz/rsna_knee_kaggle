#!/usr/bin/env bash
# unfreeze_full.sh — all twelve encoder blocks adapt, against the default six.
#
# bead: knee-p724l
# reading: unfreeze_full_s2026.holdout.npz
#
# THE FLAG LINE IS band_dense's, BYTE FOR BYTE, PLUS ONE FLAG. That is the constraint: the axis already
# holds two points measured at this exact corner (4 blocks 0.8320, 6 blocks 0.8578 midpoint), so any
# deviation here costs the comparison and buys nothing. Do not modernise this line to the shipping
# corner — the shipping corner is a different question and a 6-7 h one.
#
# THE DELTA IS READ AGAINST band_dense = 0.8578 MIDPOINT, because that is the ruler the other two points
# of the curve are on. It is not the shipping number and may not be compared to a ledger75 one.
#
# IF THIS OOMs, DROP --batch, NEVER THE BLOCK COUNT. The block count is the question being asked.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train unfreeze_full \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --unfreeze-last 12 \
  --labels "$LABELS" --out "$MODELS/unfreeze_full_s2026.pt"
