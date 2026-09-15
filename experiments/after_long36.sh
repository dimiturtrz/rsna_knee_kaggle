#!/usr/bin/env bash
# after_long36.sh — grand fragment leave-one-out probes (knee-feb62.11) in the window's last hour, after the long36 pair.
set -u
cd "$(dirname "$0")/.."
export UV_NO_SYNC=1
CLOSE=$(date -u -d "2026-09-16T04:00:00Z" +%s)
COST=900
stamp() { echo "[$(date -u +%H:%MZ)] $*"; }

until grep -q "AFTER WINDOW DONE" logs/after_window_grand.log; do sleep 60; done
for arm in probe224_grand_nomixup probe224_grand_noaug probe224_grand_nohard probe224_grand_noema; do
  if [ $(( $(date -u +%s) + COST )) -gt "$CLOSE" ]; then stamp "skip $arm: does not end before 04:00Z"; continue; fi
  stamp "start $arm"
  bash experiments/arms/$arm.sh
  stamp "end $arm"
done
stamp "LONG36 PROBES DONE"
