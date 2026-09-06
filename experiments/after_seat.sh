#!/usr/bin/env bash
# Replaces after_oof.sh's remaining queue: mix30 out, the first champion-seat retrain in.
#
# WHY. `foreign_distill_mix30`'s own pre-registration predicts a null and its teacher seat curve reads
# -0.0005 at 1x; `oof_teacher` (0.8681) and `foreign_distill_mix07` (0.8656) landing 0.0025 apart
# already answered the question mix30 was going to ask — the teacher's SOURCE is not the mechanism, so
# a third point on the ratio is a measurement of something closed. `seat_distill_lr4e5` (knee-ozpyq)
# asks the only question whose answer licenses seven hours of card: does the +0.0107 transfer to a seat
# recipe that is not band_dense.
#
# WHY A FOURTH FILE AND NOT AN EDIT. bash reads a script by byte offset as it executes. after_oof.sh's
# watcher is BLOCKED inside `bash experiments/arms/labels_v4hybrid.sh`, a child process with the trainer
# under it — killing the parent leaves the arm on the card untouched and discards only the queue behind
# it. Editing after_oof.sh in place would rewrite bytes the live shell has not read yet.
#
# THE MARKER IS SPELLED WITHOUT THE SEED and the LOG IS SPELLED WITH IT — `_mark` writes the run_train
# name while `_arm_log` names the file for the checkpoint basename, and the file lives one directory
# DOWN from the checkpoints, in $MODELS/experiments/.
set -u
cd "$(dirname "$0")/.."

MARKER="ARM DONE labels_v4hybrid"
ARM_LOG="D:/models/rsna_knee_abnormality/experiments/labels_v4hybrid_s2026.full.log"

until grep -q "$MARKER" "$ARM_LOG" 2>/dev/null; do sleep 60; done
echo "$MARKER seen — starting the reordered queue $(date -u +%FT%TZ)"

bash experiments/arms/aprior_fixed.sh
echo "QUEUE aprior_fixed exited $? $(date -u +%FT%TZ)"

bash experiments/arms/seat_distill_lr4e5.sh
echo "QUEUE seat_distill_lr4e5 exited $? $(date -u +%FT%TZ)"
