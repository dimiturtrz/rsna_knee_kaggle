#!/usr/bin/env bash
# The whole sagittal stack at the control's density: does coverage keep paying, or does it stop?
#
# WHY THIS ARM AND NOT ANOTHER BAND CELL. The span x density square is closed and it is ADDITIVE, which
# retires the fixed-budget model that motivated it. Measured on fold 0 at 4e-5 whole-stack, each main
# effect twice, at both levels of the other:
#
#     span   0.20-0.80 -> 0.10-0.90   +0.0039 at 12 slices   +0.0038 at 20 slices
#     density      12 -> 20 slices    +0.0014 at 0.20-0.80   +0.0013 at 0.10-0.90
#
# Interaction ~0.0001. Two levers, no competition between them, and each reproduces at both levels of
# the other - which is what makes a set of sub-floor cells say something a single sub-floor cell cannot.
#
# THE MECHANISM THIS ARM TESTS, and it is band_span's, not band_dense's. Widening the span at held
# density bought REGIONAL structures +0.0079 and FOCAL structures +0.0011 - ACL +0.0148, Contusion
# +0.0108, Baker's +0.0105, Effusion +0.0079, against MCL -0.0039 and Lateral Meniscus -0.0054. Adding
# slices INSIDE a span does the opposite (focal +0.0036, regional -0.0019, starring Synovitis +0.0194
# and PF OA +0.0164). So the two levers are orthogonal in their per-label populations as well as in
# their means: coverage buys the diffuse and fluid findings, density buys the focal ones.
#
# That reading makes one prediction nothing has tested. If the regional gain is COVERAGE, then covering
# the rest of the stack extends it. If instead the outer tenths are off-joint soft tissue - which is the
# stated reason SLICE_BAND excludes them (slot_reader.py:31) - the regional gain stops or reverses at
# the point the band leaves the joint. Those two are distinguishable on one arm, and only on this axis:
# every remaining band cell inside 0.10-0.90 is a knob, and knee-jk3's rule forbids running knobs.
#
# NOT FREE, and it is the first band arm that is not. images_336px_12sl.npy is the 0.20-0.80 default
# (SLICE_BAND, slot_reader.py:34) and images_336px_12sl_0.1-0.9band.npy is the wide one; 0.00-1.00 has
# never been decoded. The decode is the price of the only falsifiable question the band line has left.
#
# control  band_span = 0.8565 (0.10-0.90 at 12 slices) for the coverage delta at held density, and
#          lr4e-5_ws = 0.8525 for the total against the span axis's own origin. Same recipe, same fold.
# cost     one decode, then the control's step time (72 images a study); 12 epochs.
#
# PREDICTION, written before the run. Coverage is real but SATURATING, because the outer tenths hold
# progressively less joint: the regional rows keep the sign they took under band_span (Effusion,
# Contusion, Baker's, ACL >= 0 against band_span) while the focal rows do not recover, and the mean
# lands between 0.8565 and 0.8605 - a gain smaller than the +0.0039 the first widening bought.
# THE REFUTATION CONDITION is the regional rows turning NEGATIVE against band_span while the mean falls
# below 0.8540: that would mean the outer tenths are not weak signal but ACTIVE distraction, the band
# is a filter rather than a budget, and the correct next move is a NARROWER band than 0.20-0.80 rather
# than any wider one - the opposite of everything this line has run.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train band_full \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT --vram-gb 26 \
  --backbone-lr 4e-5 --slices 12 --band-low 0.0 --band-high 1.0 \
  --labels "$LABELS" --out "$PROCESSED/band_full_s2026.pt"
