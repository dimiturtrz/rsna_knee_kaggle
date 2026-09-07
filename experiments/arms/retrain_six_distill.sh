#!/usr/bin/env bash
# retrain_six_distill — the champion's SIX seats refitted under the out-of-fold teacher. One flag pair,
# every seat, nothing else moved: no seat swapped, no member added, vote rule and member count and
# serving precision all held.
#
# WHAT LICENSES IT. `seat_distill_lr4e5` is the same treatment on ONE seat, measured against its own
# control at the identical corner: `lr4e-5_ws` 0.8525 -> 0.8632 offline, **+0.0107**, against a
# pre-registered 0.8605 bar. That is the per-seat term this arm applies six times.
#
# ITS PRECEDENT IS `calibsix`, NOT THE PASS-THROUGH TABLE, AND THE PRECEDENT IS THE PESSIMISTIC HALF.
# `comp/champion.json` states the principle this arm is betting against: **IMPROVING EVERY SEAT AT ONCE
# DOES NOT COMPOUND.** `calibsix_v1` is the identical operation class — champion six, one flag on every
# seat, no seat moves — pre-registered at 0.946 low / 0.949 optimistic and it **read 0.941**: +0.002
# realised against +0.0065 predicted, a **31%** realisation. Two things to hold beside that 31% rather
# than against it. Its per-seat offline term was only +0.0038, so a +0.002 board delta read at
# `BOARD_RESOLVABLE` 0.0015 carries a realisation interval of roughly 15-55%. And calibration is a
# homogenising treatment on the TARGETS — the exact kind that eats vote diversity — whereas distillation
# pulls each seat toward an ORDERING; whether that is the same kind of homogenisation is the open
# question this arm answers.
#
# THE THREE BANDS, pre-registered before the card is spent, over the champion `raptorweight_v1` 0.944:
#   >= 0.953   the uniform-treatment pass-through is real. The retrain alone reaches the owner's target.
#   0.948-0.952  `calibsix` class. The term transfers but is damped; a SECOND lever is then mandatory
#                and the teacher itself is it (gen-2, five folds, born-again round two).
#   <= 0.946   diversity is eaten, the same way calibsix's was. The mitigation is a PER-SEAT teacher
#              rather than one shared ordering, and that is the next arm, not a re-run of this one.
# The arithmetic behind the middle band: 6 x +0.0107 offline x board_transfer 1.72 x 0.31 realised
# -> +0.006 -> **0.950**. Read `board_transfer` live before quoting it; this line is prose.
#
# ONE VARIABLE against `calibsix.sh` + `final_fit_six.sh`'s `calib_span_ff`: `$DISTILL`. Every other flag
# on every seat is copied VERBATIM from the arm that produced the member it replaces, which is
# `final_fit_six.sh`'s own rule — a reassembly, however sensible each flag looks alone, is a configuration
# nobody ran, and at `--fold -1` there is no holdout left to catch it.
#
# THE TEACHER MUST COVER ALL 4406 STUDIES OR THIS ARM DIES IN EPOCH ONE, AND `require` IS WHY IT DOES NOT.
# `TeacherTargets.rows` (`kneemri/training/teacher_targets.py:70`) RAISES on any study with no teacher
# row — a positive witness, per batch, not a mask. `teacher_oof_corpus.npz` covers folds 1-4 only (3498
# of 4407), because `oof_teacher`'s hole is correct for an arm SCORED on fold 0: a teacher that read the
# holdout leaks the evaluation labels into the student. This arm has no holdout. It is scored on the
# BOARD, so there is no evaluation set for a fold-0 reading to leak into, and the constraint that creates
# the hole does not reach it. `teacher_oof_full.npz` is the filled file, written to its own name so no
# fold-0-scored arm can pick it up by accident, and `require`d below so a missing teacher refuses to
# start rather than costing seven hours to discover.
#
# WHICH FILL, AND THE WITNESS DECIDED IT RATHER THAN AN ARGUMENT. `oof_teacher --fill` writes the four
# fold twins plus one fold-0 block and reports how that block reads beside them, per label, on three
# axes. The first candidate was `band_dense_s2026`, on the ground that it is the fifth read of the same
# family. It reads: mean outside the fold-to-fold scatter on 8/12 labels, calibrated on 6/12, **spread on
# 9/12 and higher every time**. Prevalence cannot move spread, so that is the reader's SHAPE and not the
# fold's contents — `band_dense` is 336 px / 20 slices and the four twins are 224 px / 12 — and it was
# refused for this arm. `oof_fold0.sh` builds the same-kind block for 25 minutes of card and is the
# teacher this arm gets; `after_seat_fold0.sh` runs it between `seat_distill` and `compose` and rebuilds
# `teacher_oof_full.npz` from it, re-running the witness on the new block. If that block ALSO reads
# outside the scatter, the heterogeneity is a SECOND variable across 8.5 h with no holdout to separate it
# afterwards, and the three bands above are then read one band wider. Nothing else about this arm changes
# either way — the teacher's FILENAME is the same, so this file needed no edit to switch blocks.
#
# COST six twelve-epoch fits at the champion's own geometries. `calibsix` measured ~425 s/epoch over
# 4,406 studies for five, about 7.1 h; six is about **8.5 h of card** and ZERO submission slots to
# produce. Reading it costs one slot and it is the only lever on the table that reaches 0.950 at all.
#
# deliberate: --distill-weight 0.7 (HELD from oof_teacher, foreign_distill_mix07 and seat_distill_lr4e5,
# deliberate: all three of which found it the best point on the weight curve; not re-searched here)
# deliberate: --teacher-targets teacher_oof5.npz (five REAL out-of-fold blocks, no fill; the corpus file
# deliberate: cannot cover a --fold -1 fit and the raise is per batch)
# deliberate: every other flag copied verbatim per seat from calibsix.sh / final_fit_six.sh
# bead: knee-ygi2k
#
# unmeasurable: five of the six seats fit at --fold -1 and retain no holdout, so no seat here can be
# unmeasurable: seat-scored, curve-scored or distance-scored. Its ruler is a board row taken with a slot.
# unmeasurable: `merged_blend_df` is fold-0 like its original and is the arm's ONLY offline read — take
# unmeasurable: it against merged_blend_cf with convergence_check first.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

MERGED_BLEND="$PROCESSED/labels/merged_blend_v1.csv"
require "$MERGED_BLEND"

# THE TEACHER IS `teacher_oof5.npz`, NOT THE FILLED FILE (changed 2026-09-07, mastermind 03:03Z). The
# fill was only ever a stand-in for a fifth block of the same kind: `oof_fold0.sh` now TRAINS that block
# at the twins' own 224/12 geometry, so all five are real out-of-fold readers and the heterogeneity
# caveat the header raises against a `band_dense` fill does not apply. It also makes seat six honest:
# `merged_blend_df` is scored on fold 0, and fold 0's teacher rows come from the twin that HELD fold 0
# out, so the leak that voided `seat_distill_lr4e5` cannot reach it (knee-zwcyk).
TEACHER="$PROCESSED/teacher_oof5.npz"
require "$TEACHER"

FIT="--fold -1 --ema"
DENSE="--slices 20 --band-low 0.1 --band-high 0.9"
CALIB="--calibrated-targets"
DISTILL="--distill-weight 0.7 --teacher-targets $TEACHER"

# SEAT SIX RUNS FIRST, and the reason is scheduling rather than modelling (mastermind 03:03Z). It stays
# FOLD-0, exactly as its original is, so it is the only seat in this arm that any offline ruler can read
# — the other five are `--fold -1` and hold nothing out. Running it first puts a reading on the board's
# own timescale: its fold-0 number against `merged_blend_cf` lands ~1.2 h in, hours before the oof5
# solo's board row, so the kill switch has evidence before the remaining seats are paid for. Nothing
# about any fit changes; six independent runs have no order.
run_train merged_blend_df \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --scheme merged $CALIB $DISTILL \
  --labels "$MERGED_BLEND" --eval-labels "$LABELS" --out "$MODELS/merged_blend_df.pt"

run_train final_ws_df \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $DENSE $CALIB $DISTILL \
  --labels "$LABELS" --out "$MODELS/final_ws_df.pt"

run_train calib_span_df \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.1 --band-high 0.9 $CALIB $DISTILL \
  --labels "$LABELS" --out "$MODELS/calib_span_df.pt"

run_train grade_ladder_df \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $DENSE --grade-ladder $CALIB $DISTILL \
  --labels "$LABELS" --out "$MODELS/grade_ladder_df.pt"

run_train through_dense_df \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $DENSE --through-plane $CALIB $DISTILL \
  --labels "$LABELS" --out "$MODELS/through_dense_df.pt"

run_train lr4e-5_ws_df \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 $CALIB $DISTILL \
  --labels "$LABELS" --out "$MODELS/lr4e-5_ws_df.pt"

