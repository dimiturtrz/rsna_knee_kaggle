#!/usr/bin/env bash
# seat_distill_clean — the same seat a third time, under a teacher that cannot leak. This is the arm
# that decides whether THE WHOLE DISTILLATION AXIS is real or is a measurement artefact, and it costs
# 43 minutes to find out.
#
# WHY IT EXISTS NOW. Two readings landed tonight and together they impeach the axis:
#   `lr4e-5_ws_s2026`          0.8525  control, no teacher
#   `seat_distill_lr4e5_s2026` 0.8633  same seat, GEN-1 teacher (`teacher_oof_corpus.npz`)   +0.0108
#   `seat_distill_gen2_s2026`  0.8640  same seat, GEN-2 teacher (`teacher_gen2_full.npz`)    +0.0007
# The gen-2 teacher's own blocks read **+0.0121** above the gen-1 blocks they replace, five folds out of
# five. A teacher that much better moved the student by +0.0007. Either the pass-through is ~6% — or the
# +0.0121 was never teacher quality at all.
#
# THE SECOND READING IS THE ONE TO TAKE SERIOUSLY, and the arithmetic is uncomfortably exact. The gen-1
# twins were fitted PLAIN. The gen-2 twins were fitted under `teacher_oof_full.npz`, whose fold-k rows
# are written by models that TRAINED on fold k — so each gen-2 twin's own holdout score is inflated by
# residual mimicry, the mechanism D30 measured at **+0.0127** (README.md:8526, knee-zwcyk). Measured
# +0.0121 against a mechanism priced at +0.0127 is not a gain with a leak in it; it is the leak.
# If that is right, `teacher_gen2_full.npz` is not a better teacher, the born-again round bought nothing,
# and the 8.5 h six-seat retrain it was built to feed has no premise.
#
# BUT IT IMPLICATES THE FIRST READING TOO, WHICH IS THE BIGGER CLAIM. `seat_distill_lr4e5`'s +0.0108 over
# its control was ALSO taken under a leaky teacher — `teacher_oof_corpus.npz` is folds 1-4, every block
# fitted without `--exclude-fold`, every one of them having trained on fold 0, and the seat is scored on
# fold 0. +0.0108 measured against the same mechanism priced at +0.0127 is the same coincidence a second
# time. **Distillation may buy nothing on this seat and may never have.** That +0.0108 is the per-seat
# term `retrain_six_distill` applies six times, it is quoted in the champion's own reasoning, and every
# distilled member in the pool carries an offline number taken the same way.
#
# THE ONE VARIABLE: the teacher file, again — `teacher_clean_full.npz`. Its own header states the
# property this arm needs: the same five blocks with folds 1-4 refitted under `--exclude-fold 0`
# (`oof_clean_fold[1-4]_s2026`) plus `oof_fold0_s2026` unchanged, 4407 rows, all 224/8, **not one of them
# written by a model that saw fold 0**. Epochs, rate, slices, band, pooling, throughput and labels are
# copied verbatim from `seat_distill_lr4e5.sh`, so this is the third point on a line with one moved
# variable and not two.
#
# PRICE: 12 epochs at 168-217 s (MEASURED — `seat_distill_gen2` ran 168 s/epoch, `seat_distill_lr4e5`
# PRICE: 217 s) = **35-43 minutes of card**, zero submission slots.
# GAIN: it prices the distillation term honestly for the first time. Every other reading of that term in
# GAIN: this repo was taken through a teacher that had seen the holdout.
# P(clears 0.8600 — i.e. distillation survives as a real treatment) ~ 45%.
# RATIO: 0.7 h to decide whether a term the champion's reasoning multiplies six times exists at all.
# There is nothing else on the table that prices that much for that little.
#
# PRE-REGISTERED BANDS, written before the card is spent:
#   >= 0.8600  distillation is real and the leak is a surcharge on top of it. Gen-2 is then simply not a
#              better teacher, the born-again axis closes on its own measurement, and `retrain_six` may
#              be re-priced off THIS number rather than off +0.0108.
#   0.8560-0.8599  distillation is real but roughly half of every distilled offline number in this repo
#              is leak. Re-price the pool; the axis survives, the magnitudes do not.
#   <= 0.8555  distillation buys nothing on this seat that the control does not already have. The
#              +0.0108 was the leak, the +0.0121 was the leak, and the whole axis is a measurement
#              artefact that has been steering card and slots for weeks. Retract in writing, re-price
#              every distilled member, and do NOT run the six-seat retrain.
# The control is 0.8525 and its own 2sd bar is 0.0071, so the bottom band is deliberately set INSIDE
# that bar: "indistinguishable from the control" is the verdict, not "below" it.
#
# KNOWN COST OF THE CLEAN TEACHER, recorded rather than argued away: starving the clean twins of one
# fold costs them −0.0067 as teachers (`compose_v4b_clean.sh:47`). So this arm is reading a WEAKER
# teacher honestly, and a result at the bottom band is therefore ambiguous between "distillation is
# artefact" and "this teacher is too weak to teach". The disambiguator is the gen-2 probe already on
# disk: a teacher +0.0121 stronger moved the student +0.0007, so teacher strength demonstrably does not
# drive this student. Both arms are needed and both now exist.
#
# deliberate: --teacher-targets teacher_clean_full.npz (the one variable)
# deliberate: --distill-weight 0.7 (HELD from every arm that has searched it; not re-searched here)
# deliberate: --epochs 12 --slices 12 --band-low 0.2 --band-high 0.8 --backbone-lr 4e-5 (verbatim from
# deliberate: seat_distill_lr4e5.sh, so the three points differ in the teacher file alone)
# bead: knee-ygi2k
# reading: seat_distill_clean_s2026.holdout.npz
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

TEACHER="$PROCESSED/teacher_clean_full.npz"
require "$TEACHER"

run_train seat_distill_clean \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --distill-weight 0.7 --teacher-targets "$TEACHER" \
  --labels "$LABELS" --out "$MODELS/seat_distill_clean_s2026.pt"
