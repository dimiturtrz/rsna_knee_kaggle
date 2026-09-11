#!/usr/bin/env bash
# after_cleanstack.sh -- hold the card until distill_stack_clean finishes, then run chainfour_v5blend.
#
# THE LOG IS NAMED FOR THE CHECKPOINT AND THE MARKER FOR THE RUN. File
# `distill_stack_clean_s2026.full.log` (basename of --out minus .pt), line inside it `ARM DONE
# distill_stack_clean` (run_train's name). Greping the checkpoint spelling in the marker waits forever
# with an idle card behind it, and queue_check reads OK the whole time.
#
# AND THE GATE IS AN OR WITH PID LIVENESS: the marker alone hangs forever on a silent exit-0 death,
# the pid alone fires the next arm the instant a crash frees the card.
source "$(dirname "$0")/lib.sh"

LOG="$MODELS/experiments/distill_stack_clean_s2026.full.log"

clean_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*distill_stack_clean*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

while true; do
  grep -aq "ARM DONE distill_stack_clean" "$LOG" 2>/dev/null && break
  clean_alive || break
  sleep 30
done
echo "=== CLEAN GATE OPEN at $(date -u +%H:%MZ) ==="

bash "$(dirname "$0")/arms/chainfour_v5blend.sh"
