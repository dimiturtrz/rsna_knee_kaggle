#!/usr/bin/env bash
# aug_long.sh — the intensity block at LENGTH, asking whether augmentation changes the SHAPE of the
# long curve rather than whether it raises a 12-epoch mean.
#
# THIS AXIS HAS A REFUTATION ALREADY, AND THIS ARM DOES NOT OVERTURN IT. `intensity_augment` scored
# 0.8459, −0.0064 against res392, behind at every epoch (ledger row `intensity_augment`). That answered
# the question it asked — does the intensity block raise a TWELVE-epoch mean at 392 px — and the answer
# is no. It is quoted here so nobody reads this arm as a silent re-run of a closed cell.
#
# WHY THE OLD TEST COULD NOT SEE THIS AXIS. Two reasons, and both are the ruler rather than the lever.
# First, the control was `res392_ws` at 392 px and the default rate — the stale operating point that
# knee-aixw is about and that `calib_span` was written up for; this arm's control is the settled one.
# Second and decisive: a regulariser is priced by what it prevents, and at twelve epochs there was
# almost nothing to prevent. `band_dense` peaks at epoch 8 and the twelfth epoch is barely off it. An
# augmentation that costs training signal early and buys invariance late is behind at every epoch of a
# schedule that ends before the cost turns into a saving, which is exactly the shape the old row
# records. Behind at every epoch of twelve is not evidence about epoch twenty.
#
# WHAT MAKES THE QUESTION LIVE NOW, and it is a measurement taken tonight rather than an argument.
# `band_dense_long` is running the same recipe to 24 epochs. Through epoch 15 it peaks at **epoch 6,
# 0.8587**, and then goes 0.8521 / 0.8507 / 0.8520 / 0.8500 / 0.8488 / 0.8499 while its train loss
# falls from 0.65 to 0.50. Holdout down 0.0099 from its own peak while the loss keeps halving is a
# memorisation signature, not a schedule that needs more epochs. The decline is LARGER than the whole
# gap the old intensity row lost by — so the thing the regulariser has to pay for is now on the table
# and was not before.
#
# THE MECHANISM IS SUPERVISION, NOT CONTRAST. `intensity_augment` argued from the manufacturer spread
# (0.8020–0.8256) and aimed at scanner invariance. That argument is unchanged and is not this one. This
# arm argues that 58 image-read labels over 4,407 studies is a supervision-limited fit, that a
# supervision-limited fit memorises once the schedule outlasts the signal, and that perturbing the
# input is the cheapest thing this repository has that makes memorising harder. The intensity block is
# reached for because it is the augmentation surface that already EXISTS, is off by default, and is
# ablatable by one flag — not because contrast is the suspected cause.
#
# IT IS ALSO THE ONLY CHEAP THING THE REGIMEN GAP ADMITS. knee-jk3 sizes the frontier at 0.054 above us
# and names its regimen — sixty epochs across twenty members — as the visible difference. We cannot buy
# sixty epochs of a curve that turns down at six. Whatever makes length productive is upstream of
# every member count; this arm asks whether the augmentation surface is enough of it to see.
#
# control  band_dense_long (24 epochs, 4e-5, 20 slices, band 0.10-0.90, max pooling, whole stack).
#          Single variable: --intensity-aug. Same schedule, same operating point, same cache.
#
# PREDICTION. The peak EPOCH moves late — 12 or beyond — and the last third of the curve is flat rather
# than declining, because that is what a regulariser does to an overfitting run. The peak MEAN clears
# band_dense_long's peak. If the mean gains, it concentrates in the REGIONAL group (Effusion,
# Synovitis, Baker's, Contusion, Fracture): diffuse findings are the ones whose evidence is spread over
# many slices, so they are the ones a memorising model can least afford to have perturbed.
#
# REFUTATION, and it is deliberately about the SHAPE so that a flat mean still answers something:
#   - the peak epoch is still at or below 10 AND the peak mean is more than 0.0062 below
#     band_dense_long's peak. Then the intensity block neither delays the turn nor raises the ceiling,
#     the old row generalises to length after all, and augmentation is retired on this corpus rather
#     than on this schedule.
# THE THIRD OUTCOME IS THE INTERESTING ONE and is enumerated on purpose: the peak moves late and the
# curve flattens while the MEAN does not rise. That would say the regulariser works and the ceiling is
# not overfitting — supervision itself is the bound, which sends the next arm to knee-mxj's label pass
# rather than to another input perturbation, and it does so on evidence instead of on preference.
#
# NO --vram-gb: DeviceBudget derives the cap from --slices, and 20 slices reserved 26.92 GB on every
# epoch of band_dense, attend_dense and focal_dense alike. Augmentation is uint8 and in-place on the
# batch, so it does not move the reservation either.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train aug_long \
  $RECIPE --epochs 24 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --intensity-aug \
  --labels "$LABELS" --out "$PROCESSED/aug_long_s2026.pt"
