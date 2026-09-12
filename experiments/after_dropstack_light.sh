#!/usr/bin/env bash
# after_dropstack_light.sh -- wait for `dropstack_dense`, then run `dropstack_light` on the freed card.
# Armed 2026-09-12T08:16Z while dropstack_dense was on epoch 9 of 12 (286 s/epoch, so ETA ~08:25Z).
#
# CONDITIONAL, UNLIKE ITS PREDECESSOR, AND THE CONDITION IS THE POINT. `after_dropstack.sh` was
# deliberately unconditional because SAM and stochastic depth are different mechanisms. These two are the
# SAME mechanism at two rates, so dropstack_dense's number decides whether this arm has a question left:
#
#   dropstack_dense >= 0.8634  it already beat the control at the FULL rate. The law is already wrong and
#                              the escalation is a proper rate sweep, not this single point. SKIP.
#   dropstack_dense <  0.8634  the discrimination this arm exists for is live. RUN.
#
# Running it in the first case would spend 56 minutes confirming something already known, which is the
# exact failure the owner named as "spamming instruments that never succeed and never teach us anything".
#
# THE LOG IS NAMED FOR THE CHECKPOINT AND THE MARKER FOR THE RUN. The file is
# `dropstack_dense_s2026.full.log` (basename of --out minus .pt) and the line inside it reads
# `ARM DONE dropstack_dense` (run_train's name). Greping the checkpoint spelling in the marker waits
# forever with an idle card behind it.
#
# THE GATE IS AN OR WITH PID LIVENESS, because the marker alone hangs forever on a silent exit-0 death
# while the pid alone fires the next arm the instant a crash frees the card.
source "$(dirname "$0")/lib.sh"

LOG="$MODELS/experiments/dropstack_dense_s2026.full.log"
CONTROL=0.8634

dense_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*dropstack_dense*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

while true; do
  grep -aq "ARM DONE dropstack_dense" "$LOG" 2>/dev/null && break
  dense_alive || break
  sleep 30
done
echo "=== CARD FREE at $(date -u +%H:%MZ) ==="

BEST=$(grep -ao 'holdout mean AUC [0-9.]*' "$LOG" 2>/dev/null | awk '{print $4}' | sort -rn | head -1)
echo "dropstack_dense best holdout mean AUC: ${BEST:-NONE} (control $CONTROL)"

if [ -z "$BEST" ]; then
  echo "NO SCORE ON DISK -- refusing to run the follow-up on an unread control."
  exit 1
fi

if awk -v b="$BEST" -v c="$CONTROL" 'BEGIN{exit !(b>=c)}'; then
  echo "SKIP: dropstack_dense cleared the control at the full rate. The half-rate point answers nothing;"
  echo "      the escalation is a rate SWEEP, which is a queue decision and not this watcher's to make."
  exit 0
fi

echo "RUN: the rate-vs-mechanism discrimination is live."
bash "$(dirname "$0")/arms/dropstack_light.sh"
