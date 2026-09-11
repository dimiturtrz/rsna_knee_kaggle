#!/usr/bin/env bash
# calib_hold_clean_distill — the project's best licence-clean corner plus the only distillation term this
# repo has ever measured honestly. Two orthogonal treatments that have never been combined, at the one
# geometry where BOTH were read against the SAME control.
#
# THE GRID, all four cells at 336 px / 20 slices / band 0.1-0.9 / 12 epochs / 4e-5, one control:
#   `band_dense_s2026`             0.8578  no teacher, no calibration            — the control
#   `calib_dense_hold_s2026`       0.8634  --calibrated-targets --hold-annotations   +0.0056
#   `foreign_distill_mix07_s2026`  0.8657  foreign teacher @0.7                      +0.0079
#   THIS ARM                          ?    calibration AND a clean in-domain teacher @0.7
# The foreign cell is the highest and it is CLOSED: all three foreign-teacher members sit in
# `comp/members.declines` on licence exposure (:677-679), so a member built there cannot ship whatever it
# reads. This arm takes the same idea through the channel that can.
#
# WHY THE DISTILLATION TERM IS NOW WORTH SPENDING CARD ON, having been retracted six hours ago. The
# retraction (README, 2026-09-11 02:25Z) killed the +0.0108 and +0.0121 readings as leak, not
# distillation itself. The honest term is now measured TWICE, at two different geometries, through two
# different clean teachers, and it replicates exactly:
#   224/12 corner: `lr4e-5_ws_s2026` 0.8525 -> `seat_distill_clean_s2026` 0.8581  (teacher_clean_full)  +0.0056
#   336/20 corner: `band_dense_s2026` 0.8578 -> `seat_clean_distill_s2026` 0.8634 (teacher_clean)       +0.0056
# Two corners, two teacher files, +0.0056 and +0.0056. That is a knob, not a step — and this arm treats
# it as exactly that: it is ADDED to the best corner, never multiplied across six seats.
#
# THE TEACHER IS `teacher_clean_full.npz` AND THE CHOICE IS LOAD-BEARING. Folds 1-4 refitted under
# `--exclude-fold 0` plus `oof_fold0_s2026` unchanged: 4407 rows, all 224/8, not one block written by a
# model that saw fold 0. `teacher_oof_corpus.npz` and `teacher_oof_full.npz` would both read ~+0.005
# higher here and every point of it would be residual mimicry against this arm's own holdout.
#
# ONE VARIABLE against `calib_dense_hold.sh`: the two distillation flags. Epochs, rate, slices, band,
# pooling, throughput, labels, `--calibrated-targets` and `--hold-annotations` are copied verbatim, so
# `calib_dense_hold_s2026` at 0.8634 is the matched control and is NOT refitted.
#
# PRICE: 12 epochs at ~361 s (MEASURED off `band_dense_s2026.metrics.jsonl`, the same corner) = **~1.2 h
# PRICE: of card**, zero submission slots. Both inputs are on disk.
# GAIN: a SHIPPABLE member at or above the project's honest solo record. `oof_teacher`'s 0.8685 — the
# GAIN: number quoted as the best solo this project owns — was itself taken through a leaky teacher and
# GAIN: is due the same ~0.005 discount, which puts the honest record near 0.8634.
# P(clears 0.8670) ~ 45%.  P(clears 0.8634, i.e. the treatment is not negative here) ~ 75%.
# RATIO: 1.2 h for a shippable member on a twice-replicated mechanism at the best corner available.
#
# PRE-REGISTERED BANDS, written before the card is spent:
#   >= 0.8690  the two treatments are ADDITIVE (0.8634 + 0.0056 = 0.8690). This is the project's best
#              honest solo by a clear margin, it is licence-clean, and it is the first member in weeks
#              with a mechanism-backed claim on the 0.874 the 0.955 target needs.
#   0.8650-0.8689  additive but damped, which is what `calibsix` did to a uniform treatment (31%
#              realisation). Still the best licence-clean solo on disk; roster it and stop treating the
#              distillation knob as composable with anything else.
#   0.8600-0.8649  the two treatments OVERLAP — calibration and a teacher are carrying the same
#              information, and the knob does not stack on a corner that already has calibrated targets.
#              That closes stacking, and it is worth knowing for 1.2 h.
#   <= 0.8599  the teacher FIGHTS the calibrated targets: the clean teacher's rows are uncalibrated
#              orderings and at weight 0.7 they overwrite the calibration this corner's +0.0056 came
#              from. Distillation is then corner-specific, not a knob, and the two-corner replication
#              above is a coincidence of two uncalibrated corners.
#
# deliberate: --distill-weight 0.7 (HELD; the settled point, not re-searched here)
# deliberate: --teacher-targets teacher_clean_full.npz (the clean teacher; a leaky one would inflate this
# deliberate: arm's own fold-0 holdout by the ~0.005 the retraction just measured)
# deliberate: --calibrated-targets --hold-annotations --slices 20 --band-low 0.1 --band-high 0.9
# deliberate: --epochs 12 --backbone-lr 4e-5 (verbatim from calib_dense_hold.sh, so one moved variable)
# bead: knee-ygi2k
# reading: calib_hold_clean_distill_s2026.holdout.npz
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

TEACHER="$PROCESSED/teacher_clean_full.npz"
require "$TEACHER"

run_train calib_hold_clean_distill \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --distill-weight 0.7 --teacher-targets "$TEACHER" \
  --labels "$LABELS" --out "$MODELS/calib_hold_clean_distill_s2026.pt"
