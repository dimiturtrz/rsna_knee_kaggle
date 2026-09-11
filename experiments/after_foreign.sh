#!/usr/bin/env bash
# Third in the chain: fires foreign_labels_dense when hold_clean_distill releases the card.
#
# The gate is the CHECKPOINT spelling in the FILE and the RUN name in the MARKER — two different
# variables, see CLAUDE.md. hold_clean_distill writes hold_clean_distill_s2026.full.log and marks it
# "ARM DONE hold_clean_distill". The wait is an OR with pid liveness: the marker alone hangs forever on
# a silent exit-0 death, the pid alone fires the next arm the instant a crash frees the card.
source "$(dirname "$0")/lib.sh"

LOG="$MODELS/experiments/hold_clean_distill_s2026.full.log"

prev_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*hold_clean_distill*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

# START GATE. hold_clean_distill has not launched yet — it is itself waiting behind
# calib_hold_clean_distill — so prev_alive is false NOW and the release loop below would break
# immediately and put this arm on a busy card. Wait for it to exist before waiting for it to finish.
while ! prev_alive && ! grep -aq "ARM DONE hold_clean_distill" "$LOG" 2>/dev/null; do sleep 30; done
echo "=== hold_clean_distill OBSERVED at $(date -u +%H:%MZ), now waiting for release ==="

while true; do
  grep -aq "ARM DONE hold_clean_distill" "$LOG" 2>/dev/null && break
  prev_alive || break
  sleep 30
done
echo "=== CARD RELEASED FOR THE LABEL ARM at $(date -u +%H:%MZ) ==="

bash "$(dirname "$0")/arms/foreign_labels_dense.sh"
