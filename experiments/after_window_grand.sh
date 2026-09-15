#!/usr/bin/env bash
# after_window_grand.sh — keeps the card busy until the window closes at 04:00Z (knee-feb62.10, knee-pqfbs).
# Waits for WINDOW DONE, then runs each long36 arm only if its measured epoch time x 36 ends before the close.
set -u
cd "$(dirname "$0")/.."
export UV_NO_SYNC=1
CLOSE=$(date -u -d "2026-09-16T04:00:00Z" +%s)
stamp() { echo "[$(date -u +%H:%MZ)] $*"; }

until grep -q "WINDOW DONE" logs/window_grand.log; do sleep 60; done
for spec in labelwise_dense_drop40_long36:6660 labelwise_ext_drop40_long36:7420; do
  arm=${spec%%:*}; cost=${spec##*:}
  if [ $(( $(date -u +%s) + cost )) -gt "$CLOSE" ]; then stamp "skip $arm: ${cost}s does not end before 04:00Z"; continue; fi
  stamp "start $arm"
  bash experiments/arms/$arm.sh
  stamp "end $arm"
done
stamp "AFTER WINDOW DONE"
