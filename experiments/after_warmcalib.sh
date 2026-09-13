#!/usr/bin/env bash
# after_warmcalib.sh — release knee-bjpq4 when `warm_calib_lowlr_drop40_long` lands.
#
# IT IS ARMED BEFORE ITS PREDECESSOR EXISTS. `after_cleancalib.sh` launches that arm at ~15:18Z, so at
# arming time neither the log nor the process is there — and a pid-liveness OR would read "not running"
# and fire this arm onto a busy card immediately. So the first phase waits for the LOG TO APPEAR, and only
# then does the marker-or-pid gate apply.
#
# The log is named for the CHECKPOINT (`warm_calib_lowlr_drop40_long_s2026.full.log`) while the MARKER TEXT
# carries the `run_train` name (`ARM DONE warm_calib_lowlr_drop40_long`). Both verified against the arm.
set -u

LOG="D:/models/rsna_knee_abnormality/experiments/warm_calib_lowlr_drop40_long_s2026.full.log"
APPEAR_LIMIT=120   # 30s polls — an hour of slack over the predecessor's ~15 remaining minutes

running() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*warm_calib_lowlr_drop40_long*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

waited=0
while [ ! -f "$LOG" ] && [ "$waited" -lt "$APPEAR_LIMIT" ]; do
  sleep 30
  waited=$((waited + 1))
done
if [ ! -f "$LOG" ]; then
  echo "=== WARM_CALIB NEVER STARTED — refusing to fire onto an unknown card at $(date -u +%H:%MZ) ==="
  exit 1
fi
echo "=== WARM_CALIB LOG APPEARED at $(date -u +%H:%MZ) ==="

while true; do
  grep -aq "ARM DONE warm_calib_lowlr_drop40_long" "$LOG" 2>/dev/null && break
  grep -aq "ARM FAILED warm_calib_lowlr_drop40_long" "$LOG" 2>/dev/null && break
  running || break
  sleep 30
done
echo "=== WARM_CALIB RELEASED at $(date -u +%H:%MZ) ==="

bash "$(dirname "$0")/arms/labelwise_depth24_negpull.sh"
