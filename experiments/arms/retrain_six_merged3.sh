#!/usr/bin/env bash
# retrain_six_merged3.sh — the six champion seats refitted on the THREE-site corpus, full-strength teacher.
#
# bead: knee-cy3wj (held: this arm runs only if noisy_student takes branch 1 or the LatMen branch)
#
# THIS ARM IS HELD, AND THE CONDITION IS PRE-REGISTERED. It costs ~8.5 h of card. It runs only if
# `noisy_student` reads >= 0.865, OR reads 0.856-0.865 WITH Lateral Meniscus recovered past
# merged3_corpus's -0.0217. Any other branch means the foreign block is still displacing our supervision
# and this arm would buy six worse seats at eight and a half hours (mastermind 04:48Z).
#
# WHAT MOVES, against retrain_six_distill. Two things: `--scheme merged3` on every seat, and a teacher
# built at FULL strength rather than starved. Everything else — the recipe, the calibrated targets, the
# 0.7 distill weight, the EMA, the twelve epochs, seat six running first and fold-0 — is character for
# character the arm it is derived from.
#
# WHY THE TEACHER MAY BE FULL STRENGTH HERE AND MAY NOT BE IN noisy_student. Five of these six seats are
# `--fold -1`: they hold nothing out, so no offline ruler reads them and the BOARD is the ruler. There is
# no holdout for a teacher to leak into. `teacher_readings --teach-all` is the deliberate spelling of
# that, it logs the consequence loudly, and `--fold -1` on its own still raises. Seat six (merged_blend_df)
# IS fold-0 and IS read offline — and it is taught by the same file, so ITS offline number is void under
# this teacher and only the board may price it. That is a change from retrain_six_distill, where seat
# six's teacher rows came from the twin that held fold 0 out. Stated here rather than discovered later.
#
# ONE RANK POPULATION, NOT TWO SHARDS. The teacher is ONE pass over the whole merged3 corpus rather than
# `teacher_oof5.npz` joined to an external-only pass. A rank is a position WITHIN a population: a shard
# ranked over our 4,406 and a shard ranked over the foreign 1,987 put a meaningless separation on every
# cross-population pair, and a separation is exactly what the distill term reads. `TeacherFile.merged`
# joins shards of ONE pass, which is a different thing.
#
# THE COST NOBODY HAS PRICED YET — READ THIS BEFORE ARMING. merged3 exists ONLY at 336/12. The six seats
# of the champion span THREE read groups (20 slices at 0.1-0.9, 12 at 0.1-0.9, 12 at 0.2-0.8), and on
# merged3 they collapse to ONE. Four of the six lose the geometry that distinguishes them and become the
# same read with different heads. `dark6` and `farseat_v1` already read flat seat-for-seat, so the entry's
# margin over a smaller one is not obviously carried by seat count — but geometry diversity is the axis
# that has NOT been closed by a board flat, and this arm spends it. If that is not acceptable, the honest
# variant is three seats on merged3 at 336/12 beside three unchanged 336/20 seats, which keeps two read
# groups and costs half the card. Decide that BEFORE the 8.5 h, not after.
#
# THE TEACHER FILE THIS ARM REQUIRES, built separately so the card cost is visible:
#
#   .venv/Scripts/python.exe -m kneemri.inference.teacher_readings \
#     --checkpoints $MODELS/oof_fold[0-4]_s2026.pt \
#     --scheme merged3 --slices 12 --band 0.2 0.8 --teach-all \
#     --out $PROCESSED/teacher_merged3_full.npz
#
# reading: merged_blend_df.holdout.npz
# unmeasurable: the five --fold -1 seats hold nothing out; the board is their only ruler, and this arm's
# unmeasurable: reading is an ENTRY built from them, not a number any of them carries.
# deliberate: --distill-weight 0.7 (the settled point on the weight curve; not re-searched)
# deliberate: --scheme merged3 on all six (the cache exists at one geometry; see the cost note above)
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

MERGED3_LABELS="$PROCESSED/labels/merged3_labels.csv"
require "$MERGED3_LABELS"

TEACHER="$PROCESSED/teacher_merged3_full.npz"
require "$TEACHER"

FIT="--fold -1 --ema"
GEOM="--slices 12 --band-low 0.2 --band-high 0.8"
CALIB="--calibrated-targets"
DISTILL="--distill-weight 0.7 --teacher-targets $TEACHER"
CORPUS="--scheme merged3 --labels $MERGED3_LABELS --eval-labels $LABELS"

# SEAT SIX RUNS FIRST for scheduling, exactly as in retrain_six_distill: it is the only fold-0 seat, so
# it is the only one that lands a number at all — void under this teacher for ranking, still able to say
# whether the fit is BROKEN. A seat that reads 0.75 is a defect; one that reads 0.87 is uninterpretable.
run_train merged_blend_dm \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $GEOM $CALIB $DISTILL $CORPUS --out "$MODELS/merged_blend_dm.pt"

run_train final_ws_dm \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $GEOM $CALIB $DISTILL $CORPUS --out "$MODELS/final_ws_dm.pt"

run_train calib_span_dm \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $GEOM $CALIB $DISTILL $CORPUS --out "$MODELS/calib_span_dm.pt"

run_train grade_ladder_dm \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $GEOM --grade-ladder $CALIB $DISTILL $CORPUS --out "$MODELS/grade_ladder_dm.pt"

run_train through_dense_dm \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $GEOM --through-plane $CALIB $DISTILL $CORPUS --out "$MODELS/through_dense_dm.pt"

run_train lr4e-5_ws_dm \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $GEOM $CALIB $DISTILL $CORPUS --out "$MODELS/lr4e-5_ws_dm.pt"
