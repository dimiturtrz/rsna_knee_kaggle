#!/usr/bin/env bash
# after_headlr_ship.sh — wait for `headlr_ship`, smoke `headlr_floor`, then put it on the card.
#
# WHY THIS SUCCESSOR IS ARMED BEFORE ITS PREDECESSOR IS READ. Normally the reading picks the next arm.
# Here it cannot pick a BETTER one than this in either branch: `headlr_floor` is the third point on the
# same curve, and a two-point head_lr read is ambiguous whichever way it lands (knee-h25fx). So the arm
# that would follow a win and the arm that would follow a loss are the same arm, and waiting to launch it
# would only buy an idle card.
#
# THE LOG IS NAMED FOR THE CHECKPOINT AND THE MARKER FOR THE RUN. File `headlr_ship_s2026.full.log`
# (basename of --out minus .pt); line inside `ARM DONE headlr_ship` (run_train's name).
#
# AND THE WAIT IS AN OR WITH PID LIVENESS, because the marker alone hangs forever on a silent exit-0
# death while the pid alone fires the next arm the instant a crash frees the card.
#
# THE SMOKE IS THE GATE. Sixty studies costs under a minute of card. A failed smoke leaves the card FREE
# and launches nothing — a refusal is the correct outcome, not an error to work around.
source "$(dirname "$0")/lib.sh"

LOG="$MODELS/experiments/headlr_ship_s2026.full.log"
SMOKE="$MODELS/smoke_headlr_floor_$(date -u +%H%M%S)"

arm_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*headlr_ship*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

echo "waiting on headlr_ship / $LOG at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
while true; do
  grep -aq "ARM DONE headlr_ship" "$LOG" 2>/dev/null && break
  arm_alive || break
  sleep 30
done
echo "card released at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

cd "$REPO" || exit 1
if UV_NO_SYNC=1 .venv/Scripts/python.exe -m kneemri.experiments.arm_check \
     --only headlr_floor --smoke "$SMOKE" --smoke-studies 60; then
  echo "SMOKE OK $(date -u +%Y-%m-%dT%H:%M:%SZ) - launching headlr_floor"
  bash "$(dirname "$0")/arms/headlr_floor.sh"
else
  echo "SMOKE FAILED - arm NOT launched, card left free $(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi
