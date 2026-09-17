#!/usr/bin/env bash
# Wait for vjepa_cold to release the card, then run sam_ship.
#
# THE GATE IS MARKER **OR** pid, never one alone. The marker is what a completed `run_train` writes;
# the pid is what a crashed one leaves behind. A marker-only gate holds the card idle forever after a
# death, and a pid-only gate fires the next arm into a card the dying process has not released.
#
# THE LOG FILE IS NAMED FOR THE CHECKPOINT AND THE MARKER TEXT IS NOT. `_arm_log` takes the basename
# of `--out`, so the file is `vjepa_cold_s2026.full.log`; `_mark` writes the `run_train` name, so the
# line is `ARM DONE vjepa_cold`. The seed is in one and never in the other.
#
# THE pid IS THE WINDOWS pid, read from `card_check`, so liveness is tested with `ps -W` and never
# with `kill -0`.
set -u
cd "$(dirname "$0")/.."

PID="${1:?usage: after_samship.sh <windows pid of vjepa_cold>}"
LOG="D:/models/rsna_knee_abnormality/experiments/vjepa_cold_s2026.full.log"

echo "waiting on vjepa_cold pid $PID / $LOG at $(date -u +%FT%TZ)"
until grep -q "ARM DONE vjepa_cold" "$LOG" 2>/dev/null || ! ps -W | grep -q "[^0-9]$PID[^0-9]"; do sleep 60; done
echo "card released at $(date -u +%FT%TZ); marker present: $(grep -qc "ARM DONE vjepa_cold" "$LOG" 2>/dev/null && echo yes || echo no)"

bash experiments/arms/sam_ship.sh
echo "QUEUE sam_ship exited $? $(date -u +%FT%TZ)"
