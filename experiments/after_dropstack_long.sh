#!/usr/bin/env bash
# after_dropstack_long.sh -- wait for `dropstack_long`, then run `labelwise_drop` on the freed card.
# Armed 2026-09-12T11:0xZ while dropstack_long was on epoch ~17 of 24 (279 s/epoch, so ETA ~11:21Z).
#
# UNCONDITIONAL, AND THAT IS AN ARGUMENT, NOT LAZINESS. `after_dropstack_light.sh` was conditional
# because both arms were the SAME mechanism at two rates, so the first one's number could retire the
# second's question. Here they are different questions: dropstack_long asks whether the BUDGET is the
# lever at the `max`-pooling corner (24 epochs vs 12), and labelwise_drop asks whether the regulariser
# TRANSFERS to labelwise pooling at the twelve-epoch budget the ledger is written in. No value of the
# first changes the second: a null on budget leaves the pooling cross untested, and a win on budget makes
# the cross MORE interesting, not less. So the card never idles on a branch that cannot fire.
#
# THE LOG IS NAMED FOR THE CHECKPOINT AND THE MARKER FOR THE RUN. File `dropstack_long_s2026.full.log`
# (basename of --out minus .pt); the line inside reads `ARM DONE dropstack_long` (run_train's name).
#
# THE GATE IS AN OR WITH PID LIVENESS, because the marker alone hangs forever on a silent exit-0 death
# while the pid alone fires the next arm the instant a crash frees the card.
source "$(dirname "$0")/lib.sh"

LOG="$MODELS/experiments/dropstack_long_s2026.full.log"
CONTROL=0.8654

long_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*dropstack_long*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

while true; do
  grep -aq "ARM DONE dropstack_long" "$LOG" 2>/dev/null && break
  long_alive || break
  sleep 30
done
echo "=== CARD FREE at $(date -u +%H:%MZ) ==="

BEST=$(grep -ao 'holdout mean AUC [0-9.]*' "$LOG" 2>/dev/null | awk '{print $4}' | sort -rn | head -1)
echo "dropstack_long best holdout mean AUC: ${BEST:-NONE} (dropstack_light $CONTROL)"
echo "    read its bands from experiments/arms/dropstack_long.sh -- this watcher does not branch on them."

echo "RUN: labelwise_drop, the fourth corner of the pooling-x-regulariser 2x2."
bash "$(dirname "$0")/arms/labelwise_drop.sh"
