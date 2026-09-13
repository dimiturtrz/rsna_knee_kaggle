#!/usr/bin/env bash
# after_cleancalib.sh — release knee-rkegi when `clean_calib_drop40_long` lands.
#
# The log is named for the CHECKPOINT (`clean_calib_drop40_long_s2026.full.log`) while the MARKER TEXT
# carries the `run_train` name (`ARM DONE clean_calib_drop40_long`) — two different variables, and only
# one of them is the checkpoint's. Both spellings verified against the arm before this was armed.
#
# The OR is the point: the marker alone hangs forever on a silent death, and the pid alone fires the
# next arm the instant a crash frees the card.
set -u

LOG="D:/models/rsna_knee_abnormality/experiments/clean_calib_drop40_long_s2026.full.log"

running() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*clean_calib_drop40_long*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

while true; do
  grep -aq "ARM DONE clean_calib_drop40_long" "$LOG" 2>/dev/null && break
  grep -aq "ARM FAILED clean_calib_drop40_long" "$LOG" 2>/dev/null && break
  running || break
  sleep 30
done
echo "=== CLEAN_CALIB RELEASED at $(date -u +%H:%MZ) ==="

bash "$(dirname "$0")/arms/warm_calib_lowlr_drop40_long.sh"
