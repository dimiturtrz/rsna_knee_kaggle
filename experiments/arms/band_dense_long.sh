#!/usr/bin/env bash
# band_dense_long.sh — the settled recipe given twice the schedule, which is the one axis knee-jk3
# conceded was worth something and nobody has run in the LONG direction.
#
# WHAT IS ACTUALLY OPEN. The decode-geometry family is closed on all four of its axes at about +0.005
# total (knee-jk3). The label-calibration lever came back NEGATIVE at the settled operating point
# (calib_span, -0.0020). The pooling operator was exonerated tonight (attend_dense: swapping the
# elementwise max for a learned gate stops the focal/regional trade and buys -0.0009, so the fixed
# budget is bounded somewhere other than its allocator). What knee-jk3 named and never tested is the
# REGIMEN: "pilkwang trains 60 epochs x 20 members, we train 24 x 5. Worth something; not worth 0.054."
# Worth something has never been measured. Our best single member trains for TWELVE.
#
# WHY THIS IS NOT schedule_length.sh RUN BACKWARDS. That arm asks whether epochs 7-12 are dead weight
# and is written to buy SPEED; it takes the 12-epoch trajectory as the ceiling and looks below it. The
# reason it cannot answer this question is the reason it gives itself: OneCycle ties the entire rate
# trajectory to the epoch count, so a 24-epoch run is not a 12-epoch run continued — it is a different
# schedule with a slower anneal and twice as many steps at high rate. Neither direction predicts the
# other, and only one of them has been run.
#
# control  band_dense = 0.8578 (336 px, 20 slices, 0.10-0.90, whole-stack, 4e-5, 12 epochs), the best
#          single member the project owns. SINGLE VARIABLE: that invocation with --epochs 24. Nothing
#          else moves — same span, same density, same pooling, same rate, same throughput flags.
# cost     twice band_dense's, about 2.5 h at its 365 s/epoch. That is the honest price and it is why
#          this runs against an otherwise idle card rather than ahead of anything.
#
# PREDICTION, written before the run. band_dense peaks at epoch 8 of 12 (0.8530 / 0.8519 / 0.8569 /
# 0.8578 / 0.8572 / ...) and is flat, not still climbing, when its schedule ends. Two readings follow
# and they are different claims. If the peak is a property of the DATA — the model has extracted what
# 12 slices over 0.10-0.90 contain — then 24 epochs lands within the 0.0062 floor of 0.8578 and the
# regimen is retired as a route, which is worth knowing because it is the last cheap axis on knee-jk3.
# If the peak is a property of the SCHEDULE — the anneal ended before the model was done — then the
# longer trajectory clears the floor, and since the frontier trains five times longer still, the axis
# scales and the next arm on it is 36 or 48 rather than another knob.
#
# THE REFUTATION CONDITION. A result at or below 0.8578 refutes the regimen as a lever at this operating
# point and closes the axis knee-jk3 left half-open, rather than leaving "train longer" as a permanent
# unfalsified excuse for the gap. A result ABOVE the floor with the peak arriving at epoch 16-20 is the
# confirming shape; a result above the floor whose peak still lands near epoch 8 is NOT — that would be
# scatter on a different seed of the same trajectory, and the epoch index is what separates the two.
#
# NO --vram-gb: DeviceBudget derives the cap from --slices, and 20 slices reserved 26.92 GB on every one
# of band_dense's twelve epochs and again on attend_dense's. The schedule does not touch memory.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train band_dense_long \
  $RECIPE --epochs 24 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --labels "$LABELS" --out "$MODELS/band_dense_long_s2026.pt"
