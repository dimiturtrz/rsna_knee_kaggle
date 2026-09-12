#!/usr/bin/env bash
# after_labelwise_drop.sh -- wait for `labelwise_drop`, then run `labelwise_long24` ONLY if it earned it.
#
# CONDITIONAL, AND THAT IS THE ARGUMENT. `after_dropstack_long.sh` was unconditional because its two
# arms asked different questions. These two ask the SAME question one step apart: if the regulariser
# did not transfer to labelwise pooling, stretching its budget tests nothing. So the gate is the
# arm's own pre-registered "real but small" floor, 0.8663.
#
# If the gate does not fire the card is left FREE deliberately -- the contraction finding (ledger
# 2026-09-12) says another member of our own pool cannot move the board at any seating, so an idle
# card beats an hour spent adding one.
#
# Log is named for the CHECKPOINT (labelwise_drop_s2026.full.log); the marker inside carries the RUN
# name (`ARM DONE labelwise_drop`). Gate is an OR with pid liveness: the marker alone hangs forever on
# a silent exit-0 death, the pid alone fires the instant a crash frees the card.
source "$(dirname "$0")/lib.sh"

LOG="$MODELS/experiments/labelwise_drop_s2026.full.log"
GATE=0.8663

drop_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*labelwise_drop*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

while true; do
  grep -aq "ARM DONE labelwise_drop" "$LOG" 2>/dev/null && break
  drop_alive || break
  sleep 30
done
echo "=== CARD FREE at $(date -u +%H:%MZ) ==="

BEST=$(grep -ao 'holdout mean AUC [0-9.]*' "$LOG" 2>/dev/null | awk '{print $4}' | sort -rn | head -1)
if [ -z "$BEST" ]; then
  echo "REFUSE: no holdout score in $LOG -- not running anything on a number that does not exist."
  exit 1
fi
echo "labelwise_drop best holdout mean AUC: $BEST (gate $GATE, control labelwise_dense 0.8613)"

if awk -v b="$BEST" -v g="$GATE" 'BEGIN{exit !(b>=g)}'; then
  echo "RUN: labelwise_long24 -- the regulariser transferred, so the budget question is worth asking here."
  bash "$(dirname "$0")/arms/labelwise_long24.sh"
else
  echo "HOLD: $BEST is below $GATE. The regulariser did not transfer to labelwise pooling, so its"
  echo "      budget tests nothing. Card left free on purpose -- see the contraction finding."
fi
