#!/usr/bin/env bash
# after_negpull.sh — re-run `warm_calib_lowlr_drop40_long` to completion once knee-bjpq4's arm lands.
#
# The first run was cut at e8 of 24 by a PC shutdown and read +0.0090 ledger75 as a FLOOR (knee-rkegi).
# Gate: the marker (`ARM DONE|ARM FAILED labelwise_depth24_negpull`, the run_train name) OR the process gone.
set -u

LOG="D:/models/rsna_knee_abnormality/experiments/labelwise_depth24_negpull_s2026.full.log"

running() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*labelwise_depth24_negpull*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

echo "waiting for 'ARM DONE labelwise_depth24_negpull' in $LOG"
until grep -aq "ARM DONE labelwise_depth24_negpull\|ARM FAILED labelwise_depth24_negpull" "$LOG" 2>/dev/null || ! running; do
  sleep 60
done
echo "released $(date -u +%H:%MZ)"
bash "$(dirname "$0")/arms/warm_calib_lowlr_drop40_long.sh"
