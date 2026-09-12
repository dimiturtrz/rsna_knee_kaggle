#!/usr/bin/env bash
# Waits for mergedk_corpus to finish, then runs its matched control base224_control (knee-vrvus).
#
# THE MARKER IS SPELLED WITHOUT THE SEED. `_mark` (lib.sh:642) writes the run_train NAME while
# `_arm_log` (:685) names the FILE for the checkpoint basename, so `--out .../mergedk_corpus_s2026.pt`
# writes `mergedk_corpus_s2026.full.log` containing the line `ARM DONE mergedk_corpus`.
#
# THE ARM LOG IS UNDER $MODELS/experiments/, NOT logs/ — lib.sh re-runs from a snapshot and the
# snapshot's tee writes there. A watcher gated on a path with no producer waits forever with the card
# idle behind it, and queue_check reads OK the whole time.
#
# The pid arm of the OR is the point: the marker alone hangs forever on a silent exit-0 death.
# `ps -W` because the MSYS pid is a different number and `kill -0` reads the wrong table.
set -u
cd "$(dirname "$0")/.."

MARKER="ARM DONE mergedk_corpus"
ARM_LOG="D:/models/rsna_knee_abnormality/experiments/mergedk_corpus_s2026.full.log"
PID="${1:-}"

while true; do
  grep -q "$MARKER" "$ARM_LOG" 2>/dev/null && { echo "marker seen $(date -u +%FT%TZ)"; break; }
  if [ -n "$PID" ] && ! ps -W | grep -q "[^0-9]$PID[^0-9]"; then
    echo "pid $PID gone without a marker $(date -u +%FT%TZ)"; break
  fi
  sleep 60
done

bash experiments/arms/base224_control.sh
echo "QUEUE base224_control exited $? $(date -u +%FT%TZ)"
