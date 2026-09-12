#!/usr/bin/env bash
# Waits for blendhold_drop40 to finish, then runs chainfour_drop40 (knee-wy4vt -> knee-uhknv).
#
# MARKER SPELLING: `_mark` writes the run_train NAME (`ARM DONE blendhold_drop40`) while `_arm_log`
# names the FILE for the checkpoint basename, so the file is blendhold_drop40_s2026.full.log and the
# marker inside it carries no seed. The log lives under $MODELS/experiments/, not logs/.
#
# The pid arm of the OR is the point: the marker alone hangs forever on a silent exit-0 death, and the
# pid alone fires the next arm the instant a crash frees the card. `ps -W` because the MSYS pid is a
# different number and `kill -0` reads the wrong table.
set -u
cd "$(dirname "$0")/.."

MARKER="ARM DONE blendhold_drop40"
ARM_LOG="D:/models/rsna_knee_abnormality/experiments/blendhold_drop40_s2026.full.log"
PID="${1:-}"

while true; do
  grep -q "$MARKER" "$ARM_LOG" 2>/dev/null && { echo "marker seen $(date -u +%FT%TZ)"; break; }
  if [ -n "$PID" ] && ! ps -W | grep -q "[^0-9]$PID[^0-9]"; then
    echo "pid $PID gone without a marker $(date -u +%FT%TZ)"; break
  fi
  sleep 60
done

bash experiments/arms/chainfour_drop40.sh
echo "QUEUE chainfour_drop40 exited $? $(date -u +%FT%TZ)"
