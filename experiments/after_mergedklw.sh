#!/usr/bin/env bash
# after_mergedklw.sh — wait for `labelwise_ext_drop40_long_ff`, then run `mergedk_labelwise_drop40_long`
# (knee-dl65h). Armed 2026-09-13 08:10Z while the ff arm was on epoch 3.
#
# NO GATE. The ff arm fits `--fold -1` and holds nothing out, so there is no number to gate on — its
# pre-registration is on the BOARD, not on an offline ruler. The mergedk arm is queued as a DIVERSITY
# SEAT, not as a solo challenger, so a weak solo does not disqualify it either.
#
# THE LOG IS NAMED FOR THE CHECKPOINT AND THE MARKER FOR THE RUN. Here they coincide
# (`--out .../labelwise_ext_drop40_long_ff.pt`, run name `labelwise_ext_drop40_long_ff`), but the gate
# is still written from the two separate sources rather than derived from one.
#
# AND THE GATE IS AN OR WITH PID LIVENESS, because the marker alone hangs forever on a silent exit-0
# death while the pid alone fires the next arm the instant a crash frees the card.
source "$(dirname "$0")/lib.sh"

LOG="$MODELS/experiments/labelwise_ext_drop40_long_ff.full.log"

ff_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*labelwise_ext_drop40_long_ff*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

while true; do
  grep -aq "ARM DONE labelwise_ext_drop40_long_ff" "$LOG" 2>/dev/null && break
  ff_alive || break
  sleep 30
done
echo "=== FF ARM RELEASED at $(date -u +%H:%MZ) ==="

bash "$(dirname "$0")/arms/mergedk_labelwise_drop40_long.sh"
