#!/usr/bin/env bash
# teacher_clean — the out-of-fold teacher REBUILT so that no teacher ever saw fold 0.
#
# WHY THE EXISTING TEACHER IS NOT OUT-OF-FOLD. `oof_fold_k` (k=1..4) was fitted at `--fold k`, which
# trains on every OTHER fold — fold 0 included. So each of the four teachers memorised fold 0's labels.
# A student distilled from their targets on folds 1-4 learns to mimic four functions that each carry
# fold 0 inside them, and is then SCORED on fold 0. The teacher never predicts fold 0; it does not have
# to. The path is function mimicry, and it inflates every distilled fold-0 number this project owns.
#
# THE BOARD IS WHAT CAUGHT IT, and it is the un-leaked ruler. Solos of this family transfer offline ->
# board at ~0.79 of the offline delta: spansolo 0.8545 -> 0.921, holdsolo 0.8634 -> 0.928. `oof_teacher`
# read 0.8681 offline -- ABOVE holdsolo -- and scored **0.926**, BELOW it. Pass-through 0.37 from the low
# anchor and negative from the high one. The offline +0.0107 is mostly not real; the board says the solo
# term is ~+0.002 at best.
#
# WHAT THIS ARM CHANGES: ONE FLAG. `--exclude-fold 0`, which drops fold 0 from TRAINING without putting
# it in the holdout (`ReportFolds.holdout`'s `excluded`). Each run then fits on two of folds 1-4, holds
# out the third of them, and has never seen fold 0 in any form. Every other flag is copied verbatim from
# `oof_folds.sh` — same 224 px, 8 epochs, 12 slices, band 0.2-0.8, same encoder — so the clean teacher
# differs from the leaky one in exactly the leak and nothing else.
#
# THE COST OF THE FLAG IS REAL AND IS NOT A CONFOUND FOR THE STUDENT. Each teacher now fits on ~2,600
# studies instead of ~3,500, so each is WORSE, and a worse teacher can only lower the student's score.
# That direction is what makes the reading safe: if the student still clears its bar with weaker
# teachers, the term survives the leak by more than this arm can explain away.
#
# ITS DELIVERABLE IS THE ARRAYS, NOT THE SCORES. Like `oof_folds.sh`, no checkpoint here is a member and
# no holdout number here is read as a recipe result. `oof_teacher --twins 'oof_clean_fold[1-4]*'`
# assembles the four blocks into the teacher `seat_clean_distill` consumes. The four blocks cover folds
# 1-4 = exactly the studies a fold-0-scored student trains on, so no fill is needed and none is honest:
# a fold-0 block in this file would be the leak re-entering by the front door.
#
# COST four runs, ~25 min each at 224/8, ~1.7 h. It buys the ruler for an 8.5 h retrain that is now NOT
# licensed, so its ratio is the retrain's whole cost against 1.7 h.
#
# deliberate: --exclude-fold 0 (the arm's entire thesis)
# deliberate: --fold 1..4, --image-size 224, 8 epochs (verbatim from oof_folds.sh; an instrument)
# reading: oof_clean_fold*_s2026.holdout.npz
# bead: knee-ygi2k
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

for fold in 1 2 3 4; do
  run_train "oof_clean_fold${fold}" \
    $RECIPE --fold "$fold" --exclude-fold 0 --epochs 8 --slice-pooling max --whole-stack $THROUGHPUT \
    --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 --image-size 224 \
    --labels "$LABELS" --out "$MODELS/oof_clean_fold${fold}_s2026.pt"
done
