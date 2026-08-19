#!/usr/bin/env bash
# The last cell of the span x density factorial: the WIDE span at the control's density.
#
# THE 2x2, all at the same 4e-5 whole-stack recipe on fold 0, all four cells from cached decodes:
#     0.20-0.80, 12 slices   lr4e-5_ws       0.8525   the control every member reads
#     0.10-0.90, 12 slices   THIS ARM                 span alone, density held
#     0.20-0.80, 20 slices   band_density    (running) density alone, span held
#     0.10-0.90, 20 slices   band_dense      0.8578   both moved
#
# WHY IT IS NOT band_wide AGAIN. band_wide is 0.10-0.90 at SIXTEEN slices: it widened the span AND kept
# mm-per-slice constant, which means it moved two things and its +0.0028 is the pair. Nothing has held
# density at the control's twelve while widening the span, so no cell of this square is a clean main
# effect yet — band_density supplies one, this arm supplies the other, and the pair supplies the
# INTERACTION, which is the quantity the fixed-budget model actually predicts. Additive halves say the
# representational budget is spent per-slice; a negative interaction says the budget is spent per-STACK
# and the two levers compete for it, which is what the symmetric per-label trades (PF OA -0.0194 wide,
# +0.0192 dense; MCL +0.0133 wide, +0.0022 dense) have been hinting at without being able to prove.
#
# WHY IT IS WORTH A SLOT ON ITS OWN (knee-nas5). Decode/band is the only ensemble axis measured to
# transfer to the board at MORE than the usual compression (0.0059 offline -> 0.014 on the board, 2.4x),
# and this member reads a pixel group — twelve slices over the wide span — that no seated member reads.
# The read is the SEAT VALUE against the pinned champion and the per-label shape, not solo AUC alone.
#
# FREE: images_336px_12sl_0.1-0.9band.npy has been on disk since 2026-08-18. No decode, no new artefact.
#
# control  lr4e-5_ws = 0.8525 for the span delta at held density, and band_dense = 0.8578 for the
#          density delta at held span. Same recipe, same fold, same label table.
# cost     the control's step time (72 images a study); 12 epochs.
#
# PREDICTION, written before the run. The fixed-budget model says max pooling spends one budget over
# whatever span it is given, so widening at UNCHANGED density trades rather than adds: the collaterals
# and off-centre compartments (MCL, Lateral/Medial Meniscus, Lateral OA) rise against the control while
# the central rows (PF OA, Synovitis) FALL, and the mean lands within noise of 0.8525 — a trade, not a
# gain. The refutation condition is a mean clearing 0.8560 with no central row losing ground: that would
# mean the outer slices carry usable signal the twelve-slice budget was not already saturated by, and
# band_wide's gain would have to be re-read as coverage rather than as the mm-per-slice it held fixed.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train band_span \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.1 --band-high 0.9 \
  --labels "$LABELS" --out "$PROCESSED/band_span_s2026.pt"
