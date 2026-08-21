#!/usr/bin/env bash
# bead: knee-40wh
# through_dense.sh — put through-plane information INTO a slice's feature vector, before the vector exists.
#
# THE ONE MECHANISM THE AGGREGATION CHAPTER LEFT OPEN, and it is named in `depth_true`'s own close. That
# chapter saturated slice aggregation from three directions and every direction agreed: more slices
# (-0.0001), more slots / 83,297 unseen slices (+0.0002), and all four combining rules within 0.0019 of
# each other. `depth_true` added the ORDER the other three were blind to and lost 0.0015. The reading
# both arms wrote was the same — the limit is UPSTREAM, in what ONE SLICE's feature vector contains —
# and every one of those arms mixed depth AFTER the encoder had already reduced each slice to a vector,
# so not one of them could have put through-plane information into it. This does, and it does it before
# any feature exists.
#
# IT IS NOT `channels`, and that distinction is the whole reason this is a new flag rather than a revival
# of the old default. Channel stacking also feeds three slices to the encoder's three input channels — but
# it does so INSTEAD of per-slice pooling: one forward per slot, no slice axis at all, which couples the
# group's physical thickness to the slice count and is measurably why raising inference slices from 12 to
# 32 cost -0.018 there. Pixel-level through-plane mixing was therefore only ever measured BUNDLED with
# the loss of the slice axis, and the bundle is what lost. Here the slice axis, the pooling, the token
# count, the parameter count and the FLOP count are all exactly the control's. The only thing that changes
# is what the three input channels hold: (i-1, i, i+1) rather than three copies of i.
#
# FREE BY CONSTRUCTION, which is why it is worth an hour. The trunk's patch-embedding convolution already
# takes three channels and was pretrained on three that DIFFER; today we hand it three identical ones and
# waste two thirds of that stem. Nothing is added, resized or initialised, so there is no new-parameter
# story to confound the read. Edges clamp rather than wrap, so the arm differs from its control on
# `slices - 2` of `slices` slices and is identical on the other two — the conservative direction.
#
# THE RISK, priced. `cnx_dense` lost 0.0585 at MATCHED TRAIN LOSS, which is this project's sharpest
# evidence that the regime is data-limited rather than capacity-limited; a stem that suddenly sees three
# times the pixels per forward is more information into a model that has repeatedly shown it cannot use
# more. That is exactly why this is framed as an INSTRUMENT: at ~40 min it either clears the floor or it
# retires the last open through-plane mechanism for an hour of card time, and the aggregation chapter can
# then be closed without an asterisk.
#
# READ IT ON, against band_dense = 0.8578 (best epoch 8), byte-identical in every other respect:
#   * ABOVE by more than 0.0062 — the slice vector WAS the bottleneck, the whole depth axis was measured
#     at the wrong stage, and a genuine 3D stem (which costs parameters) becomes worth its price.
#   * ROUGHLY 0.8578 — the trunk cannot use through-plane context at the pixel level either, and the
#     aggregation chapter closes with its own exclusion clause discharged.
#   * BELOW — the neighbours are ACTIVE NOISE at this stem: two thirds of each patch embedding now carry
#     a different slice's anatomy, and the encoder's pretrained colour-channel structure is being asked
#     to mean depth. That is a positive finding about the stem, and it says a 3D encoder must be
#     PRETRAINED on volumes rather than adapted at its first convolution.
#
# control  band_dense = 0.8578 (best epoch 8). Same cache, band, slices, schedule, rate and pooling.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train through_dense \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --through-plane \
  --labels "$LABELS" --out "$PROCESSED/through_dense_s2026.pt"
