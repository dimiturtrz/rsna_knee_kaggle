#!/usr/bin/env bash
# Replaces after_dissent2.sh's remaining queue, per mastermind 21:42Z: arm A AHEAD of mix30.
#
# WHY A THIRD FILE AND NOT AN EDIT. bash reads a script by byte offset as it executes, so editing
# after_dissent2.sh under a live process changes what it runs next. Its watcher (pid 1376149) is
# currently BLOCKED inside `bash experiments/arms/oof_teacher.sh`, which is a CHILD process
# (1382281) with the trainer under it — killing the parent leaves the arm on the card untouched and
# only discards the queue that followed it. That is exactly the surgery wanted here: keep the run,
# replace the order.
#
# ORDER. labels_v4hybrid (knee-wko44 arm A) first: the label lever has a board precedent and it is the
# label-table axis's only card-priced test. `aprior_fixed` (knee-nin6h) second, 73 min, an instrument
# whose result is an argument. `foreign_distill_mix30` last — its own pre-registration predicts a null
# and its teacher seat curve already reads -0.0005 at 1x, so slipping it to tomorrow costs nothing.
#
# THE MARKER IS SPELLED WITHOUT THE SEED and the LOG IS SPELLED WITH IT. `_mark` writes the run_train
# name (`ARM DONE oof_teacher`) while `_arm_log` names the file for the checkpoint basename. And the
# arm log does not live under logs/ here, and it is not beside the checkpoints either: `run_train` tees
# it into $MODELS/experiments/, one directory DOWN from where the .pt files sit. The first arming of
# this watcher guessed $MODELS/oof_teacher_s2026.full.log and would have waited forever with the card
# idle behind it — the third spelling hazard on one marker, after the missing seed and the logs/ guess.
# `ls $MODELS/experiments/*.full.log` before arming; do not derive the path from the checkpoint's.
set -u
cd "$(dirname "$0")/.."

MARKER="ARM DONE oof_teacher"
ARM_LOG="D:/models/rsna_knee_abnormality/experiments/oof_teacher_s2026.full.log"

until grep -q "$MARKER" "$ARM_LOG" 2>/dev/null; do sleep 60; done
echo "$MARKER seen — starting the reordered queue $(date -u +%FT%TZ)"

bash experiments/arms/labels_v4hybrid.sh
echo "QUEUE labels_v4hybrid exited $? $(date -u +%FT%TZ)"

bash experiments/arms/aprior_fixed.sh
echo "QUEUE aprior_fixed exited $? $(date -u +%FT%TZ)"

bash experiments/arms/foreign_distill_mix30.sh
echo "QUEUE mix30 exited $? $(date -u +%FT%TZ)"
