#!/usr/bin/env bash
# rampsix — the champion's five own-corpus seats refitted with the curriculum RAMP, and nothing else.
# bead: knee-fxgm
#
# THIS IS A UNIFORM-LIFT ARM, WHICH IS THE ONE MEMBER-SIDE MECHANISM THE BOARD HAS NOT FLATTENED.
# Seven board flats closed the SEATING axis (reweight, add, swap, count, precision, pooling lens,
# distance-with-quality) and every one of them moved WHICH members sit. This arm moves none of that:
# same six seats, same vote, same kit. It changes what each member IS by one recipe flag applied to all
# of them at once, which is the shape `calibsix` already measured on this exact pool — +0.0038 pooled
# solo went to +0.0056 and +0.0084 ensembled on two nested pools. A uniform lift does NOT pay the 23%
# seat pass-through that a single better member pays (`calib_dense_hold`: +0.0056 solo, +0.0013 seated),
# because the ambiguity term is preserved when every member moves the same way.
#
# THE DOSE, AND WHERE IT WAS MEASURED. The 2x2 in the ledger (2026-09-02) closed with `disagreed_ramp`
# at 0.8574 against `disagreed_uniform`'s 0.8507, and `certainty_ramp` at 0.8584 against the same
# control. Two unrelated difficulty definitions, both ramped over 6 of 12 epochs, both +0.007; the same
# signal applied FLAT costs -0.0036. So the schedule is the effect and the signal is nearly free, and
# the dose here is ~1.8x the one calibsix carried.
#
# PRE-REGISTERED, BEFORE ANY OF IT RUNS:
#   - per-member solo: +0.004 to +0.008 over each seat's calibsix original. Below +0.002 on the first
#     two members finished, this arm is WRONG about transfer and the remaining three are not run.
#   - offline ensemble over the champion's own six (the four refits plus the two untouched seats):
#     +0.004 or better, which is above the 0.0015 the board can resolve and above the 0.0062 holdout
#     floor only in aggregate. A board slot is NOT pre-committed here; the offline ensemble read is the
#     gate, and it is read with the same six-member serving the champion uses, not a fresh search.
#
# WHY SEAT FIVE IS NOT IN HERE. `merged_blend_cf` trains on the 5,143-row merged corpus with its own
# label table, and `train_spread_honest12.npz` is indexed by the 4,407-row own-corpus training set. A
# spread array whose length disagrees with the sampler's index is not a recipe difference, it is an
# off-by-corpus bug that would train and log a plausible curve, so that seat is left exactly as it is
# and the arm reports a lift over five of six rather than pretending to six.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

SPREAD="$PROCESSED/train_spread_honest12.npz"
require "$SPREAD"

FIT="--fold -1 --ema"
DENSE="--slices 20 --band-low 0.1 --band-high 0.9"
CALIB="--calibrated-targets"
RAMP="--sample-spread $SPREAD --sample-alpha 1.0 --sample-ramp 6"

run_train final_ws_ramp \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $DENSE $CALIB $RAMP \
  --labels "$LABELS" --out "$MODELS/final_ws_ramp.pt"

run_train through_dense_ramp \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $DENSE --through-plane $CALIB $RAMP \
  --labels "$LABELS" --out "$MODELS/through_dense_ramp.pt"

run_train grade_ladder_ramp \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $DENSE --grade-ladder $CALIB $RAMP \
  --labels "$LABELS" --out "$MODELS/grade_ladder_ramp.pt"

# The geometry is SPELLED rather than inherited, exactly as calibsix spells it: the original was fitted
# on the parser defaults, so a silent default change would move this seat and not its control.
run_train lr4e-5_ws_ramp \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 $CALIB $RAMP \
  --labels "$LABELS" --out "$MODELS/lr4e-5_ws_ramp.pt"

run_train calib_span_ramp \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 $CALIB $RAMP \
  --labels "$LABELS" --out "$MODELS/calib_span_ramp.pt"
