#!/usr/bin/env bash
# Queue after `v6tears_span`: the non-tear per-column correction, then the depth instrument.
#
# REPLACES the tail of `after_v6.sh`, whose watcher shell (pid 38364) was killed WITHOUT //T while
# `v6tears_span` was mid-flight - the arm is a separate pid and a non-tree kill leaves it running,
# which is the only way to reorder a queue without costing the card the arm already on it. A live
# bash process reads its script by byte offset, so after_v6.sh was never edited.
#
# The reorder is by RATIO: `v7nontear_span` is a measured label correction (703 cells, Contusion
# wholesale at 0.833 plus the OA family gated, D352) and `labelwise_depth24` is ~80 min of 5090 for
# a pre-registered flat.
#
# The gate is the FILE's checkpoint spelling with the MARKER's run_train name, OR'd against pid
# liveness so a silent death frees the card instead of hanging the queue.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="D:/models/rsna_knee_abnormality/experiments/v6tears_span_s2026.full.log"
CONTROL_PID="${1:-45332}"

while true; do
  if grep -q "ARM DONE v6tears_span" "$LOG" 2>/dev/null; then break; fi
  if ! ps -W | grep -q "[^0-9]$CONTROL_PID[^0-9]"; then break; fi
  sleep 30
done

bash "$HERE/arms/v7nontear_span.sh"
bash "$HERE/arms/labelwise_depth24.sh"
