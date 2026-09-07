#!/usr/bin/env bash
# merged3_tranche_one.sh — the THREE 12-slice champion seats refitted on the three-site corpus.
#
# bead: knee-cy3wj (held: runs only if noisy_student takes branch 1 or the LatMen branch)
#
# THIS ARM IS HELD, AND THE CONDITION IS PRE-REGISTERED. ~4.2 h of card. It runs only if
# `noisy_student` reads >= 0.865, OR reads 0.856-0.865 WITH Lateral Meniscus recovered past
# merged3_corpus's -0.0217. Any other branch and the canonical `retrain_six_distill` runs instead.
#
# WHY THREE AND NOT SIX. merged3 exists ONLY at 336/12. The champion's six seats span THREE read groups
# (20 slices at 0.1-0.9, 12 at 0.1-0.9, 12 at 0.2-0.8); moving all six to merged3 collapses them to ONE,
# and four of the six stop being different reads. Every axis this project has closed is a SELECTION axis
# — seating, weighting, re-lensing, seven board flats. Geometry diversity has NOT been closed by any of
# them, and a six-seat merged3 retrain spends it blind for 8.5 h.
#
# So: the three seats whose read geometry ALREADY matches the cache move, and the three 336/20 seats stay
# on the canonical distilled path (`_df`, retrain_six_distill). Two read groups survive, the corpus delta
# lands on half the entry, and — the reason this ordering is right rather than merely cautious — the
# moved seats change ONE thing. A 336/20 seat moved to merged3 would carry a corpus delta and a geometry
# delta at once, and nothing downstream could tell them apart. These three carry corpus only.
#
# TRANCHE TWO IS A BOARD DECISION, NOT A PLAN. Collapsing the remaining three onto merged3 is gated by
# the 17:50Z entry (3 merged3 + 3 canonical): >= 0.946 licenses it, flat says geometry diversity was
# load-bearing and it never runs. A slot spent as a gate (mastermind 04:53Z).
#
# THE THREE SEATS, NAMED SO THE 17:50Z ASSEMBLER DOES NOT RE-DERIVE THEM:
#   merged_blend_dm   (fold 0, 12 slices, 0.2-0.8)  <- the only readable seat, see the teacher note
#   calib_span_dm     (fold -1, 12 slices, 0.1-0.9)
#   lr4e-5_ws_dm      (fold -1, 12 slices, 0.2-0.8)
# and their three canonical partners, unchanged, from retrain_six_distill:
#   final_ws_df, grade_ladder_df, through_dense_df (all 336/20 at 0.1-0.9)
#
# WHAT THE THREE SEATS ARE WORTH, STATED BEFORE THE ENTRY IS ASSEMBLED. `calib_span_dm` and
# `lr4e-5_ws_dm` differ on merged3 by BAND ALONE — 0.1-0.9 against 0.2-0.8 over the same twelve cached
# slices. On the canonical path they also differed in what the cache held. It is a real difference and it
# is kept, but it is thinner than the seat names suggest: this tranche is closer to two and a half
# distinct reads than three, and an entry built from it should not be read as though it had three.
#
# ONE RANK POPULATION. The full-strength teacher is ONE `--teach-all` pass over all 6,393 studies with
# the five `oof_fold[0-4]` twins — not `teacher_oof5.npz` joined to an external-only pass. A rank is a
# position WITHIN a population: two shards ranked over two populations put a meaningless separation on
# every cross-population pair, and ~45% of merged3's pairs are cross-population. That separation is
# exactly what `Objective.pairs` reads. `TeacherFile.merged` joins shards of ONE pass; it is not a join
# of two rankings.
#
# WHY SEAT ONE KEEPS ITS RULER AT NO EXTRA CARD COST. The two `--fold -1` seats hold nothing out, so the
# board is their ruler and the full-strength file is free of consequence for them. `merged_blend_dm` IS
# fold-0 and IS read offline, so a teacher that saw fold 0 would void its number — and it is the arm's
# only offline read. It therefore takes `teacher_merged3.npz`, the four fold-0-clean twins, which
# `after_oof5_student.sh` ALREADY BUILDS for noisy_student. No second clean pass is needed: that file was
# built at `--fold 0`, so it covers exactly the studies a fold-0 student trains on, which is all
# `TeacherFile.aligned` requires. The cost is the -0.0067 starvation on that one seat and zero card.
#
# reading: merged_blend_dm.holdout.npz
# unmeasurable: calib_span_dm and lr4e-5_ws_dm are --fold -1 and hold nothing out; the board prices them
# unmeasurable: as an ENTRY, and no number either one carries is a reading.
# deliberate: --distill-weight 0.7 (the settled point on the weight curve; not re-searched)
# deliberate: seat one takes the STARVED teacher while the other two take the full one (see above)
# control  merged_blend_cf = 0.8597 (the same seat, same geometry, merged2 corpus, no distill)
#
# THE TWO TEACHER FILES. The starved one is built by the live chain; the full one is this arm's own
# prerequisite, ~15 min over a predecoded array (no pydicom), NOT part of noisy_student's chain:
#
#   .venv/Scripts/python.exe -m kneemri.inference.teacher_readings \
#     --checkpoints $MODELS/oof_fold[0-4]_s2026.pt \
#     --scheme merged3 --slices 12 --band 0.2 0.8 --teach-all \
#     --out $PROCESSED/teacher_merged3_full.npz
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

MERGED3_LABELS="$PROCESSED/labels/merged3_labels.csv"
require "$MERGED3_LABELS"

STARVED="$PROCESSED/teacher_merged3.npz"
FULL="$PROCESSED/teacher_merged3_full.npz"
require "$STARVED"
require "$FULL"

CALIB="--calibrated-targets"
CORPUS="--scheme merged3 --labels $MERGED3_LABELS --eval-labels $LABELS"

# SEAT ONE RUNS FIRST because it is the only one that lands a number: a seat reading 0.75 is a defect
# worth killing the other two over, and the other two cannot say so about themselves.
run_train merged_blend_dm \
  $RECIPE --fold 0 --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --distill-weight 0.7 --teacher-targets "$STARVED" \
  $CALIB $CORPUS --out "$MODELS/merged_blend_dm.pt"

run_train calib_span_dm \
  $RECIPE --fold -1 --ema --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.1 --band-high 0.9 \
  --distill-weight 0.7 --teacher-targets "$FULL" \
  $CALIB $CORPUS --out "$MODELS/calib_span_dm.pt"

run_train lr4e-5_ws_dm \
  $RECIPE --fold -1 --ema --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --distill-weight 0.7 --teacher-targets "$FULL" \
  $CALIB $CORPUS --out "$MODELS/lr4e-5_ws_dm.pt"
