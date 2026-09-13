#!/usr/bin/env bash
# after_densedrop.sh — wait for `labelwise_dense_drop40_long` (knee-vs06j), then run
# `labelwise_drop_drop40_long` (knee-fnhbs). Armed 2026-09-13 10:04Z, with the dense arm on epoch 0.
#
# WHY IT IS ARMED BESIDE THE ARM AND NOT AFTER IT. The card sat idle from 09:11Z to 10:02Z because the
# previous arm's successor was paced off an ESTIMATED landing time instead of an armed watcher. Arming
# the successor at the same moment as the arm is the fix, and it costs nothing when the gate is an OR.
#
# NO GATE ON THE DENSE ARM'S NUMBER, DELIBERATELY. The two arms ask different questions: the dense arm
# asks whether the operator pair transfers to a third base, this one asks whether upgraded bases keep
# ADDING seats (measured near-additive at two: +0.0020, +0.0017, +0.0031 together). A flat dense solo
# would not answer the additivity question, so it must not block it.
#
# THE LOG IS NAMED FOR THE CHECKPOINT AND THE MARKER FOR THE RUN NAME — two separate variables, written
# here from their two separate sources rather than derived from one (lib.sh:580 vs :642).
#
# AND THE GATE IS AN OR WITH PID LIVENESS, because the marker alone hangs forever on a silent exit-0
# death while the pid alone fires the next arm the instant a crash frees the card.
source "$(dirname "$0")/lib.sh"

LOG="$MODELS/experiments/labelwise_dense_drop40_long_s2026.full.log"

dense_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*labelwise_dense_drop40_long*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

while true; do
  grep -aq "ARM DONE labelwise_dense_drop40_long" "$LOG" 2>/dev/null && break
  dense_alive || break
  sleep 30
done
echo "=== DENSE ARM RELEASED at $(date -u +%H:%MZ) ==="

bash "$(dirname "$0")/arms/labelwise_drop_drop40_long.sh"
