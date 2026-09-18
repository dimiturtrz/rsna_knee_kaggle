#!/usr/bin/env bash
# after_headlrff.sh — wait for `headlr_floor`, smoke `headlr_ship_ff`, then put it on the card.
#
# WHY THIS ARM AND NOT A READING-PICKED ONE. knee-2s0gp's shortlist is empty and every knob axis is
# closed, so there is nothing to pick; refit is the only positive measured board term the repo owns
# (knee-33iwi). `headlr_floor`'s own reading is a head_lr SHAPE, board-invisible either way, so it
# cannot choose a better successor than this in either branch — waiting on it buys only an idle card.
#
# THE ONE READING THAT CAN KILL THIS ARM is kneemri-headlrsolo (ref 56316099), the board reading of this
# member's half-fit twin. At <= 0.939 the solo transfer line bends down above offline75 0.904 and this
# arm's projection drops ~0.004. That reading lands while the arm runs; kill it by hand on that branch.
#
# THE LOG IS NAMED FOR THE CHECKPOINT AND THE MARKER FOR THE RUN. File `headlr_floor_s2026.full.log`
# (basename of --out minus .pt); line inside `ARM DONE headlr_floor` (run_train's name).
#
# AND THE WAIT IS AN OR WITH PID LIVENESS, because the marker alone hangs forever on a silent exit-0
# death while the pid alone fires the next arm the instant a crash frees the card.
source "$(dirname "$0")/lib.sh"

LOG="$MODELS/experiments/headlr_floor_s2026.full.log"
SMOKE="$MODELS/smoke_headlr_ship_ff_$(date -u +%H%M%S)"

arm_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*headlr_floor*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

echo "waiting on headlr_floor / $LOG at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
while true; do
  grep -aq "ARM DONE headlr_floor" "$LOG" 2>/dev/null && break
  arm_alive || break
  sleep 30
done
echo "card released at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

cd "$REPO" || exit 1
if UV_NO_SYNC=1 .venv/Scripts/python.exe -m kneemri.experiments.arm_check \
     --only headlr_ship_ff --smoke "$SMOKE" --smoke-studies 60; then
  echo "SMOKE OK $(date -u +%Y-%m-%dT%H:%M:%SZ) - launching headlr_ship_ff"
  bash "$(dirname "$0")/arms/headlr_ship_ff.sh"
else
  echo "SMOKE FAILED - arm NOT launched, card left free $(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi
