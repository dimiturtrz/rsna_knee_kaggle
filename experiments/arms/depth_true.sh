#!/usr/bin/env bash
# bead: knee-tqn
#
# attend_true's invocation with --slice-pooling depth substituted for attend, and NOTHING else. One
# variable, and the control is an arm that finished hours ago rather than one this queue has to pay for.
#
# WHY THIS IS THE ONLY SURVIVING VEHICLE FOR THE 3D MECHANISM. A meniscal tear or a cruciate disruption
# is CONTINUOUS across adjacent slices; a partial-volume artefact is not. Telling them apart means
# reading depth as an ORDERED sequence. Every slice pooling this project has measured is
# permutation-blind -- amax over slices, the arithmetic mean, and `attend`'s per-slice independent
# scoring all give the same answer if the slices are shuffled. `depth` is a depthwise Conv1d of span 3,
# no bias, identity-tap initialised, with slots folded into the batch so the kernel cannot cross a slot
# boundary: it is the one operator in the tree that can express continuity at all.
#
# The OTHER vehicle is closed, which is what moved this arm up the queue. knee-ec0p proposed teaching
# through-plane structure to the self-supervised objective instead -- hide a slice, predict it from its
# neighbours. Its blocker read out twice: encoder_adapt at lr 1e-4 gave 0.6902 and at lr 1e-5 gave
# 0.6875, 0.0027 apart from two adaptations whose blocks 0-5 differ FOURFOLD in drift. What both share
# is block 11 -- the block the head reads -- moved 0.55 and 0.21 of its own norm against a supervised
# maximum anywhere of 0.0080. Masked reconstruction reshapes the read block whatever it masks, and
# twelve epochs of fitting cannot walk that back. So the trunk cannot be taught depth; only the pool can.
#
# WHAT THE CONTROLS ARE, and there are three of them because this axis finally has a clean one.
#   attend_true  0.8476  -- the same recipe with a trained but ORDER-BLIND gate. The direct control.
#   slice_attention 0.8473 -- the same recipe with an UNTRAINED gate, i.e. the arithmetic mean.
#   wsfold0_e6   0.8457  -- the same recipe with elementwise max.
# Those three span 0.0019 against a 0.0062 floor, so the order-blind family is measured flat and any
# movement here belongs to order rather than to weighting.
#
# THE CLAIM THAT IS BIGGER THAN THIS ARM'S MEAN. Every null on the slice-COUNT axis was taken under an
# order-blind pool: knee-ezz measured 20 cached slices against 12 at -0.0001, extended_slots put 83,297
# genuinely unseen slices in front of the model for +0.0002, slices16_ws was -0.0005. A model with no
# way to express continuity across depth has no use for more depth and would post exactly those numbers
# whether or not the depth carried signal. If this arm reads positive, that whole axis was measured
# without the mechanism that makes it readable and re-running it becomes licensed.
#
# PREDICTION, written before the run. The four labels the 58-study vision partition says we fail on are
# exactly the continuity ones -- Lateral Meniscus -0.193, ACL -0.123, MCL -0.096, Medial Meniscus
# -0.094, each 15-30x the floor. If continuity is the mechanism those four go positive against
# attend_true and the mean lands 0.8500-0.8560. Recovering a THIRD of that deficit on four of twelve
# labels is 4/12 x 0.04 = +0.013 on the mean, just under the 1.5% bar, so the bar needs either a
# better-than-third recovery or spillover to the other eight.
#
# REFUTATION: the mean inside attend_true's floor WITH the four continuity labels flat. That says depth
# carries no extra signal at this slice density, closes the aggregation stage for good -- count axis
# included, since the operator excuse is spent -- and sends the next arm to the label/loss side. A flat
# mean with the four labels MOVING is the third outcome and is not a refutation: it would say order is
# real and is being paid for elsewhere, which is a redistribution worth an ensemble seat.
#
# cost 12 epochs at ~250 s, about 52 minutes. The Conv1d is span 3 over 12 slices and adds no
# measurable VRAM over `attend`, so the derived cap is the control's.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train depth_true \
  $RECIPE --epochs 12 --slice-pooling depth --whole-stack $THROUGHPUT \
  --backbone-lr 8e-6 --slices 12 --band-low 0.2 --band-high 0.8 \
  --labels "$LABELS" --out "$PROCESSED/depth_true_s2026.pt"
