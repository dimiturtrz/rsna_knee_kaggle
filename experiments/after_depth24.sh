#!/usr/bin/env bash
# Queue after `labelwise_depth24`: the deletions-only lone-dissent table (D363).
#
# NOT an edit to `after_v7.sh`. That watcher is live and reading its own file by byte offset, so
# editing it would shift the offsets underneath a running shell. This is a second watcher chained
# behind the arm that one is currently running, which is the only safe way to extend a live queue.
#
# The gate is the FILE's checkpoint spelling with the MARKER's run_train name, OR'd against pid
# liveness so a silent death frees the card instead of hanging the queue.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="D:/models/rsna_knee_abnormality/experiments/labelwise_depth24_s2026.full.log"
CONTROL_PID="${1:-70540}"

while true; do
  if grep -q "ARM DONE labelwise_depth24" "$LOG" 2>/dev/null; then break; fi
  if ! ps -W | grep -q "[^0-9]$CONTROL_PID[^0-9]"; then break; fi
  sleep 30
done

bash "$HERE/arms/v2delonly_span.sh"
