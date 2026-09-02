#!/usr/bin/env bash
# band224.sh — the best geometry we have, read at 224 px instead of 336.
#
# bead: knee-w84q
#
# WHY THIS ARM. `native224` measured the pixel axis as flat (0.8502 against 0.8525) at ONE geometry:
# 12 slices, 0.20-0.80 band. Its own write-up says that is what it measured and no more. Our best solo
# member is `band_dense` = 0.8578, which reads 20 slices over a wider 0.10-0.90 band, and the flat
# reading has never been checked there. This arm checks it, and the check is worth having for two
# separate reasons.
#
# THE FIRST REASON IS A PRICE. If 224 is flat here too, every future arm on the settled BEST geometry
# costs ~160 s/epoch instead of ~480 — a twelve-epoch arm in half an hour rather than an hour and a
# half — and the whole remaining schedule runs three times as wide. That is the same throughput
# argument `native224` made, applied to the geometry we would actually build a member on.
#
# THE SECOND REASON IS A REAL RISK TO THE FIRST. 20 slices over a wider band samples MORE of the stack
# and therefore samples further into its ends, where the anatomy is smaller and the structures are
# thinner. If the extra pixels are carrying anything anywhere, that is where. A flat `native224` and a
# DOWN `band224` would say the pixel axis interacts with the sampling geometry, which is a finding about
# our preprocessing rather than about DINOv2, and it would immediately un-cheapen every 224 arm queued
# behind it.
#
# ONE VARIABLE against the control: `--image-size 224`. Everything else is byte-identical to
# `band_dense`, including the batch, the rate and the twelve epochs.
#
# COST. ~160 s/epoch derived from native224's 94 at 12 slices (20/12 = 1.67x), so ~32 min for twelve
# epochs against band_dense's ~90. Memory scales the same way from native224's 8.1 GB: ~13.5 GB.
#
# THE FALSIFIABLE PART, written before the run.
#   * within +/- 0.006 of 0.8578. The pixel axis is flat at BOTH geometries, 224 becomes the default
#     read for every arm from here, and the schedule triples in width. This is the expected outcome.
#   * <= -0.015. The pixel axis interacts with the sampling geometry: the wider band needs the pixels.
#     That un-cheapens every queued 224 arm INCLUDING base224, and the follow-up is 336 at 20 slices
#     with the band narrowed, to separate "the ends need pixels" from "more slices need pixels".
#   * >= +0.015. 224 is BETTER at the wider band, which no mechanism here predicts. Suspect the harness
#     first — the position-embedding interpolation is the only thing that changes, and a gain from
#     removing it at 20 slices but not at 12 wants an explanation before it wants a member.
#
# HARNESS CHECK. The printed recipe must name "image_size": "224", "slices": "20", "band_low": "0.1",
# "band_high": "0.9". The 224/20-slice cache must exist or the run builds it first; check the log's
# opening lines before timing anything.
#
# control  band_dense = 0.8578 (best epoch 8), byte-identical recipe at 336 px.
#
# deliberate: --image-size 224 (the arm's thesis)
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train band224 \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --image-size 224 \
  --labels "$LABELS" --out "$MODELS/band224_s2026.pt"
