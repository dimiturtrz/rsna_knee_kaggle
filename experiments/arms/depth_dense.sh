#!/usr/bin/env bash
# depth_dense.sh — can the model see that a finding CONTINUES into the next slice?
#
# THE GAP, found by reading the model rather than the ledger. The data is 3D and every aggregation this
# project has ever run over the slice axis is permutation-invariant:
#   over_depth      amax within depth regions, then .mean() ACROSS regions — symmetric under region swap
#   SliceAttention  attention-MIL: the gate scores each slice from that slice ALONE
# Whole-stack reading, three encoders, two slice poolings, a resolution sweep and every decode arm sat
# on top of one of those two operators. So no arm in this project has ever tested through-plane ORDER,
# and attend_dense's +0.0002 over max did not test it either — it swapped one order-blind operator for
# another. A tear reads as a tear largely because it persists across consecutive slices, and the model
# has never been able to represent that.
#
# WHY THIS IS NOT THE ENCODER AXIS AGAIN. That axis closed on measurement: v3base_dense -0.0057 solo and
# -0.0007 as a seat, cnx_dense -0.0585 solo and -0.0042 as a seat — a second ViT family AND a CNN trunk
# outside the family entirely, both inside the noise as members. The ledger's own consequence was that
# the next member has to come from the data side rather than the encoder side. All three encoders fed
# the SAME order-blind aggregator, so that closure says nothing about this.
#
# control  band_dense = 0.8578 (336 px, 20 slices, 0.10-0.90, whole-stack, 4e-5, 12 epochs), the best
#          single member the project owns, and attend_dense = 0.8580 on the same recipe. SINGLE
#          VARIABLE: --slice-pooling depth in place of attend. Epochs held at 12 because one-cycle is
#          defined over total_steps and an epoch change would move the rate trajectory as well.
# cost     one 12-epoch fit at band_dense's step time, plus a depthwise conv of span 3 over the slice
#          axis: 3 taps per channel, a few thousand parameters against the 64-wide gate already there.
#          The conv is initialised to the IDENTITY TAP, so step zero computes exactly what attend_dense
#          computed and every difference is something training chose to add.
# price    ~1.1 h of card. Plausible gain if through-plane continuity is real supervision the model
#          currently cannot use: 1-3%. P(clears the 1.5% bar): ~25%. Priced before queueing, per
#          knee-qxse.
#
# PREDICTION, written before the run. If through-plane continuity carries signal, depth beats attend by
# more than the 0.0062 floor, and the gain should concentrate in the GRADED findings — a meniscus tear
# or a cartilage defect is defined by extent across slices, where an effusion is not. A mean that moves
# with no per-label structure behind it is a number to distrust.
#
# THE REFUTATION CONDITION, and it is worth as much. A null here says the aggregator was never the
# bottleneck: with the encoder axis already closed on three members, that points the remaining budget at
# supervision (58 of 4,407 studies carry an image-read label) rather than at architecture, and it
# retires the whole family of "read the stack better" ideas rather than leaving another cell to run.
# It does NOT kill order-awareness in general — absolute depth position is a separate claim this arm
# deliberately does not supply, so a null leaves that re-testable.
#
# NOT QUEUED, 2026-08-21, and both reasons are measurements rather than scheduling.
#   1. THE CONTROL THIS HEADER NAMES DOES NOT EXIST. `attend_dense`'s gate was outside every optimiser
#      group (knee-7gyw), so 0.8580/−0.0009 is MEAN pooling, and `depth` against `band_dense` would vary
#      two things at once: a trained gate AND the neighbour mixing this arm is about. `attend_true` runs
#      first to supply the one-variable control.
#   2. The cheap tier reads it near zero. On cached features — where `head_trial` has always trained the
#      aggregator — max 0.7605, attend 0.7632, depth 0.7625. Neighbour mixing on top of a trained gate is
#      −0.0007 there. That does not license 1.1 h of card; a frozen encoder is a weak witness for a
#      through-plane mechanism (its features were never fit to carry continuity), so this is a DEFERRAL
#      on price, not a refutation, and `DepthAttention` stays a re-testable asset.
#
# NO --vram-gb: DeviceBudget derives the cap from --slices. 20 slices reserve 28.26 GB.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train depth_dense \
  $RECIPE --epochs 12 --slice-pooling depth --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --labels "$LABELS" --out "$MODELS/depth_dense_s2026.pt"
