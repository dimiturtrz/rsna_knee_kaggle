#!/usr/bin/env bash
# The fourth corner of the span x density square: the CONTROL's span at band_dense's density.
#
# WHAT THE SQUARE HOLDS, all at the same 4e-5 whole-stack recipe on fold 0:
#     span 0.20-0.80, 12 slices   lr4e-5_ws     0.8525   the control every other member reads
#     span 0.10-0.90, 16 slices   band_wide     0.8553   widened at the control's mm-per-slice
#     span 0.10-0.90, 20 slices   band_dense    0.8578   widened AND redensified - best single member
#     span 0.20-0.80, 20 slices   THIS ARM               density alone, span held
#
# WHY THE MISSING CORNER IS THE ONE WORTH RUNNING. band_dense moved TWO things at once and is credited
# with +0.0053 over the control; band_wide isolates the span half at +0.0028. The density half has
# never been run alone, so band_dense's decomposition rests on a subtraction (0.0053 - 0.0028) that
# assumes the two effects add. This measures it directly. If density alone lands near +0.0025, the
# additive reading holds and the two halves are separable levers; if it lands at zero, band_dense's
# gain is the SPAN and the extra slices are along for the ride; if it lands near +0.0053, the span
# bought nothing and band_wide's collateral gains came from somewhere the band model does not name.
#
# WHY IT IS A MECHANISM AND NOT A THIRD POINT ON A KNOB. band_wide and band_dense settled that max
# pooling spends a FIXED representational budget over whatever span it is given, and that the trade is
# SYMMETRIC: widening bought the collaterals at the centre's expense (PF OA -0.0194, MCL +0.0133) and
# redensifying bought the centre back at the collaterals' expense (PF OA +0.0192, MCL +0.0022), each
# with intervals clearing zero. Held at the control's span, density adds slices WITHOUT changing what
# anatomy is covered - so the budget model predicts it is nearly free of the trade, which is the one
# prediction that separates 'the budget is fixed' from 'more slices always help'.
#
# WHY IT IS WORTH A CARD SLOT EVEN IF THE MEAN IS FLAT (knee-nas5). The decode/band group is the only
# ensemble axis measured to transfer to the board at MORE than the usual compression - collapsing three
# members onto one pixel group cost 0.0059 offline and 0.014 on the board, 2.4x - and a card-free greedy
# over 22 same-split members picks band_dense, band_wide and depth_regions as its first three seats,
# worth +0.0070 before any other axis is chosen. This member reads a pixel group no seated member reads.
# The read is the SEAT VALUE against the shipped four and the per-label shape, not solo AUC alone.
#
# FREE: images_336px_20sl.npy has been on disk since 2026-08-11. No decode, no new artefact - which is
# why this corner runs and 0.30-0.70 at 20 slices (a ~60 GB build that must run with no trainer beside
# it) does not.
#
# control  lr4e-5_ws = 0.8525 for the density delta at held span, and band_dense = 0.8578 for the span
#          delta at held density. Both are the same recipe on the same fold and labels.
# cost     ~ band_dense's step time; 12 epochs.
#
# PREDICTION, written before the run. If the fixed-budget reading is right: the mean lands between
# 0.8525 and 0.8578, the OA and central rows (PF OA, Medial OA, Synovitis) rise against the control
# while MCL and ACL stay within their intervals of it - density inside an unchanged span cannot buy the
# collaterals, because it does not reach them. The refutation condition is a COLLATERAL gain: MCL or ACL
# clearing its interval here would mean extra slices help by something other than covering anatomy the
# span already covers, and both band arms would have to be re-read.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train band_density \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 \
  --labels "$LABELS" --out "$PROCESSED/band_density_s2026.pt"
