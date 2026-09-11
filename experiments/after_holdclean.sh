#!/usr/bin/env bash
# after_holdclean.sh — wait for `calib_hold_clean_distill` (the eighth cell of the 336/20 factorial) and
# then run `hold_clean_distill` (the seventh, the only other unrun one). Armed 2026-09-11 04:33Z while
# the eighth cell was on epoch 3.
#
# UNGATED ON PURPOSE. The seventh cell is an INSTRUMENT: it decides whether `--hold-annotations` is a
# calibration partner or a general soft-target enabler, and that question is worth the same whatever the
# eighth cell reads. A gate here would only couple two independent readings.
#
# THE LOG IS NAMED FOR THE CHECKPOINT AND THE MARKER FOR THE RUN — file
# `calib_hold_clean_distill_s2026.full.log`, line `ARM DONE calib_hold_clean_distill`, no seed in the
# string. And the wait is an OR with pid liveness: the marker alone hangs forever on a silent exit-0
# death, the pid alone fires the next arm the instant a crash frees the card.
source "$(dirname "$0")/lib.sh"

LOG="$MODELS/experiments/calib_hold_clean_distill_s2026.full.log"

prev_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*calib_hold_clean_distill*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

while true; do
  grep -aq "ARM DONE calib_hold_clean_distill" "$LOG" 2>/dev/null && break
  prev_alive || break
  sleep 30
done
echo "=== EIGHTH CELL RELEASED at $(date -u +%H:%MZ) ==="

bash "$(dirname "$0")/arms/hold_clean_distill.sh"
