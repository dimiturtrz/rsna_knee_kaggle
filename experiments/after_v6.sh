#!/usr/bin/env bash
# Queue after `v5blend_span`: the tear correction first, the depth instrument behind it.
#
# REPLACES the tail of `after_v3.sh`, which owed `labelwise_depth24` on the same marker. The reorder is
# by RATIO, not momentum: depth24 is ~80 min of 5090 for a pre-registered flat, and `v6tears_span` is
# the largest label correction the project has measured (D350 - 277/277 contested tear cells, v2 0.000
# against cluster 1.000, over 223 distinct studies). `after_v3.sh` is killed rather than edited: a live
# bash process reads its script by byte offset, so editing one under itself corrupts what it runs next.
#
# The gate is the FILE's checkpoint spelling with the MARKER's run_train name, and it is OR'd against
# pid liveness so a silent death frees the card instead of hanging the queue.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="D:/models/rsna_knee_abnormality/experiments/v5blend_span_s2026.full.log"
CONTROL_PID="${1:-64080}"

while true; do
  if grep -q "ARM DONE v5blend_span" "$LOG" 2>/dev/null; then break; fi
  if ! ps -W | grep -q "[^0-9]$CONTROL_PID[^0-9]"; then break; fi
  sleep 30
done

bash "$HERE/arms/v6tears_span.sh"
bash "$HERE/arms/labelwise_depth24.sh"
