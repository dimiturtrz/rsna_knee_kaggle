#!/usr/bin/env bash
# Waits for merged3_oversample to finish, then runs mricore_rate (D194 §4's learning-rate arm).
#
# THE MARKER IS SPELLED WITHOUT THE SEED. `_mark` (experiments/lib.sh:642) writes the run_train NAME,
# while `_arm_log` (:685) names the FILE for the checkpoint basename via ARM_LOG_NAME (:580). So this
# arm's `--out .../merged3_oversample_s2026.pt` writes `merged3_oversample_s2026.full.log` containing
# the line `ARM DONE merged3_oversample` — the file carries the seed and the marker does not. Format
# confirmed against markers already on disk (`ARM DONE weight_average`, `ARM DONE whole_stack`).
#
# THE ARM LOG IS UNDER $MODELS/experiments/, NOT logs/. `lib.sh` re-runs an arm from a snapshot and the
# snapshot's tee writes there; `logs/` holds the `foreign_run` arms and misc. A watcher gated on a path
# with no producer waits forever with the card idle behind it, and queue_check reads OK the whole time.
# The path below was ls'd before this file was written.
#
# The pid arm of the OR is the point: the marker alone hangs forever on a silent exit-0 death, and the
# card would sit idle behind a green-looking watcher. `ps -W` because the MSYS pid is a different
# number from the Windows one and `kill -0` reads the wrong table.
set -u
cd "$(dirname "$0")/.."

MARKER="ARM DONE merged3_oversample"
ARM_LOG="D:/models/rsna_knee_abnormality/experiments/merged3_oversample_s2026.full.log"
PID="${1:-}"

while true; do
  grep -q "$MARKER" "$ARM_LOG" 2>/dev/null && { echo "marker seen $(date -u +%FT%TZ)"; break; }
  if [ -n "$PID" ] && ! ps -W | grep -q "[^0-9]$PID[^0-9]"; then
    echo "pid $PID gone without a marker $(date -u +%FT%TZ)"; break
  fi
  sleep 60
done

bash experiments/arms/mricore_rate.sh
echo "QUEUE mricore_rate exited $? $(date -u +%FT%TZ)"
