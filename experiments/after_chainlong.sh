#!/usr/bin/env bash
# Waits for chainfour_drop40_long, then runs blendhold_drop40_long (knee-kiss9).
#
# MARKER SPELLING: _mark writes the run_train NAME; _arm_log names the FILE for the checkpoint basename.
# So --out .../chainfour_drop40_long_s2026.pt writes chainfour_drop40_long_s2026.full.log containing
# the line `ARM DONE chainfour_drop40_long`. The log is under $MODELS/experiments/, not logs/.
#
# The pid arm of the OR is the point: the marker alone hangs forever on a silent exit-0 death.
set -u
cd "$(dirname "$0")/.."

MARKER="ARM DONE chainfour_drop40_long"
ARM_LOG="D:/models/rsna_knee_abnormality/experiments/chainfour_drop40_long_s2026.full.log"
PID="${1:-}"

while true; do
  grep -q "$MARKER" "$ARM_LOG" 2>/dev/null && { echo "marker seen $(date -u +%FT%TZ)"; break; }
  if [ -n "$PID" ] && ! ps -W | grep -q "[^0-9]$PID[^0-9]"; then
    echo "pid $PID gone without a marker $(date -u +%FT%TZ)"; break
  fi
  sleep 60
done

bash experiments/arms/blendhold_drop40_long.sh
echo "QUEUE blendhold_drop40_long exited $? $(date -u +%FT%TZ)"
