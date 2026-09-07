#!/usr/bin/env bash
# Replaces after_compose.sh: `oof_fold0` goes in between seat_distill and compose, and the teacher is
# rebuilt from its block before anything downstream can read the old one.
#
# WHY. `oof_teacher --fill`'s witness refused the `band_dense` fold-0 block: outside the fold-to-fold
# scatter on 9/12 labels on SPREAD, always sharper, and prevalence cannot move spread. band_dense is
# 336 px / 20 slices against the four twins' 224 px / 12, so that is reader shape. `oof_fold0` builds
# the same-kind block for 25 min of card, and `retrain_six_distill` then moves one variable instead of
# two across 8.5 h with no holdout to separate them afterwards.
#
# WHY HERE AND NOT BEFORE THE RETRAIN. Same 25 minutes wherever it sits in a card that is busy either
# way, but here the new block's witness reads at ~01:50Z with four hours to react if it ALSO sits
# outside the scatter. Before the retrain it would read at 06:15Z with the retrain waiting on it.
#
# WHY A NEW FILE AND NOT AN EDIT. bash reads a script by byte offset as it executes; after_compose.sh's
# watcher is live in its `until` loop and editing it would rewrite bytes that shell has not read yet.
# Its watcher (pid 1431500) is killed as this one is armed — one live process per owed arm, which is
# what queue_check asserts.
#
# THE MARKER IS SPELLED WITHOUT THE SEED and the LOG IS SPELLED WITH IT — `_mark` writes the run_train
# name while `_arm_log` names the file for the checkpoint basename, and the file lives one directory
# DOWN from the checkpoints, in $MODELS/experiments/. Both copied verbatim from after_compose.sh, which
# had them right.
set -u
cd "$(dirname "$0")/.."

MARKER="ARM DONE seat_distill_lr4e5"
ARM_LOG="D:/models/rsna_knee_abnormality/experiments/seat_distill_lr4e5_s2026.full.log"
PROCESSED="D:/data/volumetric/mri/processed/rsna_knee_abnormality"

until grep -q "$MARKER" "$ARM_LOG" 2>/dev/null; do sleep 60; done
echo "$MARKER seen — starting the fold-0 twin $(date -u +%FT%TZ)"

bash experiments/arms/oof_fold0.sh
echo "QUEUE oof_fold0 exited $? $(date -u +%FT%TZ)"

# The teacher is rebuilt from the new block, and --fill re-runs the witness on it. If this block also
# sits outside the scatter the line says so and the retrain's header is read one band wider; the file
# is written either way, because a retrain that cannot start is worse than one whose teacher is noted.
.venv/Scripts/python.exe -m kneemri.evaluation.oof_teacher \
  --fill oof_fold0_s2026.holdout.npz --out "$PROCESSED/teacher_oof_full.npz"
echo "QUEUE teacher_oof_full rebuilt $? $(date -u +%FT%TZ)"

bash experiments/arms/compose_v4b_oof.sh
echo "QUEUE compose_v4b_oof exited $? $(date -u +%FT%TZ)"
