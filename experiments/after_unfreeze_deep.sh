#!/usr/bin/env bash
# after_unfreeze_deep.sh — wait for `unfreeze_deep`, smoke `headlr_ship`, then put it on the card.
#
# WHY THIS ARM AND NOT THE NEXT DEPTH POINT. `unfreeze_deep`'s branches all resolve to a DESIGN question:
# a rising gradient buys a batch-matched pair at twice the wall-clock, a flat closes the axis. Neither
# names a queued successor. `headlr_ship` is owed whatever that arm reads, because it moves a different
# leaf at a different corner (knee-xuarh: the shipping line names --head-lr nowhere, so the head runs 25x
# its trunk, and the corpus' one isolating pair on that key was taken at 224/max and never here).
#
# THE LOG IS NAMED FOR THE CHECKPOINT AND THE MARKER FOR THE RUN. File `unfreeze_deep_s2026.full.log`
# (basename of --out minus .pt); line inside `ARM DONE unfreeze_deep` (run_train's name).
#
# AND THE WAIT IS AN OR WITH PID LIVENESS, because the marker alone hangs forever on a silent exit-0
# death while the pid alone fires the next arm the instant a crash frees the card.
#
# THE SMOKE IS THE GATE. Sixty studies costs under a minute of card and is what caught `unfreeze_full`
# asking 36727 MiB of a 30991 MiB card at 19:00Z today. A failed smoke leaves the card FREE and launches
# nothing — a refusal is the correct outcome, not an error to work around.
source "$(dirname "$0")/lib.sh"

LOG="$MODELS/experiments/unfreeze_deep_s2026.full.log"
SMOKE="$MODELS/smoke_headlr_$(date -u +%H%M%S)"

arm_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*unfreeze_deep*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

echo "waiting on unfreeze_deep / $LOG at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
while true; do
  grep -aq "ARM DONE unfreeze_deep" "$LOG" 2>/dev/null && break
  arm_alive || break
  sleep 30
done
echo "card released at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

cd "$REPO" || exit 1
if UV_NO_SYNC=1 .venv/Scripts/python.exe -m kneemri.experiments.arm_check \
     --only headlr_ship --smoke "$SMOKE" --smoke-studies 60; then
  echo "SMOKE OK $(date -u +%Y-%m-%dT%H:%M:%SZ) - launching headlr_ship"
  bash "$(dirname "$0")/arms/headlr_ship.sh"
else
  echo "SMOKE FAILED - arm NOT launched, card left free $(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi
