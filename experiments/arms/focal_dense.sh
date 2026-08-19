#!/usr/bin/env bash
# focal_dense.sh — enlarge the pooled budget instead of re-spending it, at the operating point we ship.
#
# THE READING THIS TESTS. Four arms now trade one structure group for the other rather than gaining on
# both: band_span buys regional (+0.0079) for focal (+0.0011), band_density buys focal (+0.0036) for
# regional (-0.0019), calib_span buys focal (+0.0037) for regional (-0.0099), and crop80_span did it
# too. A fixed pooled budget being re-allocated is the shape that produces exactly that. attend_dense
# asks whether the budget is the OPERATOR (the elementwise max over slices). This arm asks the other
# half of the same question — whether it is the WIDTH — by concatenating a third part to the pooled
# vector rather than changing how the existing parts are combined.
#
# WHY IT IS NOT A RERUN. focal_pool measured this exact pooling change on 2026-08-16 and scored 0.8432
# against whole_stack's 0.8457: -0.0025, inside the floor, flat. That reading is at the PRE-SETTLEMENT
# operating point — 12 slices, the 0.20-0.80 band, the default encoder rate — and calib_span has just
# shown what that is worth: `calibrated_targets` measured +0.0038 there, holds a seat in the 0.928
# champion package on that number, and is NEGATIVE at the recipe we actually ship. A lever's sign is a
# property of its operating point (knee bead filed 2026-08-19), and this is the first arm written to
# act on that rather than to record it.
#
# THE MECHANISM ALSO SAYS THE OLD READING WAS TAKEN WHERE IT COULD NOT SHOW. The focal tail exists to
# rescue a response that a mean over 576 tokens divides away. How much there is to rescue scales with
# how much is being pooled — and this operating point pools 20 slices over 0.10-0.90 against that one's
# 12 over 0.20-0.80. Same argument that put attend_dense ahead of slice_attention's +0.0016.
#
# control  band_dense = 0.8578 (336 px, 20 slices, 0.10-0.90, whole-stack, 4e-5, 12 epochs), the best
#          single member the project owns. SINGLE VARIABLE: that invocation with --pool cls_mean_focal
#          in place of the default. Epochs held at 12 — one-cycle is defined over total_steps, so an
#          epoch change would move the rate trajectory alongside the pooling.
# cost     one 12-epoch fit at band_dense's step time. The third part widens the head's input, not the
#          encoder, so the encoder cost is unchanged.
#
# PREDICTION, written before the run. If the budget is the WIDTH, the focal group gains ABSOLUTELY —
# not merely less negatively than the diffuse group — and the diffuse group does not pay for it. The
# named riser is Lateral Meniscus, the largest model-side gap in the corpus (-0.1932, knee-dtk) and a
# 1-3 mm finding, with Fracture second (it read +0.020 in the old arm, directional on one crossing).
#
# THE REFUTATION CONDITION, and it is the half worth having. If this trades AGAIN — focal up, regional
# down, whatever the mean does — then widening the pooled vector does not enlarge the budget either,
# and with attend_dense answering the operator half, the fixed-budget reading is refuted on BOTH of its
# axes. That retires re-allocation as an explanation outright and moves the bottleneck to supervision
# or to the encoder, which are different beads and cost different arms. A mean above the control WITH
# the trade still present is the case that most needs catching: it reads as a win and means the
# opposite. The old arm's own per-label read is the warning — its clause "half-held" only because the
# diffuse group lost, not because the focal group gained, and that is a trade wearing a win's clothes.
#
# NO --vram-gb: DeviceBudget derives the cap from --slices. 20 slices reserve 28.26 GB, which is what
# band_dense's own run reproduced. The focal tail is a head-side concatenation and does not move it.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train focal_dense \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --pool cls_mean_focal \
  --labels "$LABELS" --out "$PROCESSED/focal_dense_s2026.pt"
