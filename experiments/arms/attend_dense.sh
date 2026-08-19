#!/usr/bin/env bash
# attend_dense.sh — is the fixed pooled budget the MAX, or the model?
#
# THE OBSERVATION THIS ARM IS BUILT ON, and it is three arms wide rather than one. Every decode change
# measured at the settled operating point re-spends one pool rather than enlarging it:
#   band_span     regional +0.0079   focal   +0.0011   (widening buys the ends)
#   band_density  focal    +0.0036   regional -0.0019  (density buys the centre back)
#   calib_span    focal    +0.0037   regional -0.0099  (calibration buys the centre, costs the ends)
# Three different levers, on three different axes, all producing the same shape: one group gains and
# the other pays. A single elementwise max over the pooled slices is exactly the operator that would
# produce it — it keeps whichever slice shouted loudest per feature and discards which slice that was,
# so two structures competing for the same feature cannot both be represented. band_dense's own
# interpretation named this as dilution and called it SYMMETRIC; symmetric is what a fixed budget
# looks like from inside.
#
# WHY THE OLD MEASUREMENT DOES NOT SETTLE IT. slice_attention scored 0.8473, +0.0016 over wsfold0_e6 —
# directional, inside the 0.0062 floor, and taken at the OLD operating point: 12 slices over the
# 0.20-0.80 band at the default encoder rate. The max discards little when there is little to discard.
# The mechanism above says its cost SCALES with how much is pooled, so the lever has never been read
# where the reading predicts it matters. This is the same stale-operating-point trap calib_span just
# fell into — the calibration paid +0.0038 at the recipe of that week and came back NEGATIVE at ours.
#
# control  band_dense = 0.8578 (336 px, 20 slices, 0.10-0.90, whole-stack, 4e-5, 12 epochs), the best
#          single member the project owns. SINGLE VARIABLE: that invocation with --slice-pooling attend
#          in place of max. Epochs are held at 12 for the same reason slice_attention held them: one-
#          cycle is defined over total_steps, so an epoch budget change would move the rate trajectory
#          as well as the aggregation.
# cost     one 12-epoch fit at band_dense's step time plus one 64-wide gate; the gate's output layer is
#          zero-initialised, so training starts at the mean and has to earn any deviation.
#
# PREDICTION, written before the run. If the budget is the MAX, attention beats band_dense AND the two
# structure groups come back BOTH non-negative — the point is not the mean, it is that the trade stops.
# The gain should also be larger than the +0.0016 the same lever bought at 12 slices in a narrow band,
# because there is more being discarded here for it to recover.
#
# THE REFUTATION CONDITION, and it is the valuable half. If the arm trades AGAIN — one structure group
# up and the other down, whatever the mean does — then the budget is not the pooling operator, and the
# re-allocation reading that three arms have now suggested is refuted at its root. That retires the
# whole family of decode re-allocations rather than leaving another cell to run, and it says the
# bottleneck is capacity or supervision, which are different beads. A mean above the control WITH the
# trade still present is the case that most needs catching: it would read as a win and mean the
# opposite.
#
# NO --vram-gb: DeviceBudget derives the cap from --slices. 20 slices reserve 28.26 GB, which is what
# band_dense's own run reproduced.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train attend_dense \
  $RECIPE --epochs 12 --slice-pooling attend --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --labels "$LABELS" --out "$PROCESSED/attend_dense_s2026.pt"
