#!/usr/bin/env bash
# Kill an arm that has stopped training and started spilling.
#
# WHY THIS EXISTS
#   A shared card has no admission control. Our arm reserves its working set at launch, and a peer that
#   starts AFTERWARDS pushes the total past the device: on Windows WDDM the allocator does not OOM, it
#   spills to host memory and the arm keeps running at a fraction of its rate. Measured 2026-09-21 on
#   basinsoup_wide: epoch 0 took 1789 s against 226 s for the same recipe on a clear card, a 7x
#   slowdown, twice in one night for ~2.5 h of card that bought nothing.
#
#   Every signal anything else reads was green throughout -- 100% utilisation, a correct loss curve, a
#   correct holdout, a correct peak-memory line, no error, exit code 0 pending. A spill is a MEMORY
#   fault wearing a COMPUTE cost, so the only honest detector is throughput. This watches the one number
#   that moves.
#
#   It guards the CARD, not the result: it may only kill, never promote, rank or restart.
#
# USAGE
#   bash experiments/spill_guard.sh <full.log path> <arm pid> [slow_factor]
#
#   The log is the arm's own full.log -- named for the CHECKPOINT, not the arm, and living under
#   $MODELS/experiments/ for a run_train arm (see CLAUDE.md). The pid is the WINDOWS pid of the trainer,
#   which is what card_check prints; an MSYS pid is a different number and `kill -0` on it lies.
set -u

LOG="$1"
ARM_PID="$2"
SLOW="${3:-3}"

# The first epoch of a run is the baseline. It is measured on whatever the card looked like at launch,
# so a run that was ALREADY spilling at epoch 0 calibrates against a spilled baseline and is not caught
# -- that case belongs to the launch gate in wait_vram_then_wide.sh, which holds until the peer is gone.
BASE=""

epoch_seconds() {
  grep -oE '^epoch [0-9]+:.*\(([0-9]+)s\)$' "$LOG" 2>/dev/null |
    sed -E 's/.*\(([0-9]+)s\)$/\1/'
}

while ps -W | grep -q "[^0-9]${ARM_PID}[^0-9]"; do
  TIMES=$(epoch_seconds)
  if [ -n "$TIMES" ]; then
    [ -z "$BASE" ] && BASE=$(echo "$TIMES" | head -1)
    LAST=$(echo "$TIMES" | tail -1)
    if [ -n "$BASE" ] && [ "$BASE" -gt 0 ] && [ "$LAST" -gt $((BASE * SLOW)) ]; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SPILL: epoch ${LAST}s vs baseline ${BASE}s (>${SLOW}x)"
      nvidia-smi --query-gpu=memory.free --format=csv,noheader
      echo "killing arm pid $ARM_PID -- the card is contended, not slow"
      powershell.exe -NoProfile -Command "Stop-Process -Id $ARM_PID -Force -ErrorAction SilentlyContinue"
      exit 3
    fi
  fi
  sleep 60
done

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) arm pid $ARM_PID exited; guard stands down"
