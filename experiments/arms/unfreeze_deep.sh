#!/usr/bin/env bash
# unfreeze_deep.sh — seven encoder blocks adapt, against the default six.
#
# bead: knee-p724l
# reading: unfreeze_deep_s2026.holdout.npz
#
# THE FLAG LINE IS band_dense's, BYTE FOR BYTE, PLUS ONE FLAG. The axis already holds two points measured
# at this exact corner (4 blocks 0.8320, 6 blocks 0.8578 midpoint), so any deviation costs the comparison
# and buys nothing. Do not modernise this line to the shipping corner — that is a different, 6-7 h question.
#
# SEVEN AND NOT TWELVE BECAUSE BATCH MAY NOT MOVE. VRAM is linear at ~1299 MiB per unfrozen block:
# twelve asked 36727 MiB and nine asked 32830 against 30991 free of a 32607 MiB card (smokes,
# 2026-09-17T19:00Z and 19:03Z). The only lever that buys the difference is --batch, and RECIPE carries
# --rank-weight 1.0 whose pairwise population IS the batch (head_trial.py:231) — so a batch trim makes
# the arm measure two knobs and demands its own matched control at twice the wall-clock. Seven is the
# deepest point that fits with the batch untouched, on the ruler the other two points already use.
#
# IT IS ONE BLOCK AND THAT IS STILL WORTH ASKING, because the measured segment RISES: 4 -> 0.8320 and
# 6 -> 0.8578 is +0.0129 per block. This asks whether that gradient survives past six. A flat says the
# curve has peaked and the axis is closed; the small step is the cost of the batch staying fixed.
#
# IF THIS OOMs, LOWER THE BLOCK COUNT, NEVER --batch. The batch is what keeps the comparison single-knob.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train unfreeze_deep \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --unfreeze-last 7 \
  --labels "$LABELS" --out "$MODELS/unfreeze_deep_s2026.pt"
