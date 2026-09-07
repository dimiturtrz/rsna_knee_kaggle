#!/usr/bin/env bash
# Queues the composition arm behind `seat_distill_lr4e5`, per mastermind 23:43Z.
#
# WHY A FIFTH FILE AND NOT AN EDIT TO after_seat.sh. That shell is BLOCKED inside
# `bash experiments/arms/aprior_fixed.sh` and has not yet read the bytes past its own current line;
# bash reads a script by byte offset as it executes, so appending to it would rewrite bytes the live
# shell is about to read. Same reason after_seat.sh exists rather than an edit to after_oof.sh.
#
# THE MARKER IS SPELLED WITHOUT THE SEED, THE LOG FILE WITH IT. `_arm_log` names the file from the
# basename of --out; `_mark` writes the run_train name. Both spellings verified against the arm:
# `seat_distill_lr4e5.sh` writes `--out $MODELS/seat_distill_lr4e5_s2026.pt` under run_train name
# `seat_distill_lr4e5`.
set -u
cd "$(dirname "$0")/.."

MARKER="ARM DONE seat_distill_lr4e5"
ARM_LOG="D:/models/rsna_knee_abnormality/experiments/seat_distill_lr4e5_s2026.full.log"

until grep -q "$MARKER" "$ARM_LOG" 2>/dev/null; do sleep 60; done
echo "$MARKER seen — starting the composition arm $(date -u +%FT%TZ)"

bash experiments/arms/compose_v4b_oof.sh
echo "QUEUE compose_v4b_oof exited $? $(date -u +%FT%TZ)"
