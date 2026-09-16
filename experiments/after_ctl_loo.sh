#!/usr/bin/env bash
# after_ctl_loo.sh — the knee-rn9py control leave-one-out sweep, seven 12-epoch 224px probes back to back.
set -u
cd "$(dirname "$0")/.."
export UV_NO_SYNC=1
CLOSE=$(date -u -d "2026-09-16T04:00:00Z" +%s)
stamp() { echo "[$(date -u +%H:%MZ)] $*"; }
for arm in ctl_nodrop ctl_maxpool ctl_nocalib ctl_norank ctl_nofloor ctl_nowhole ctl_fastbb; do
  if [ $(( $(date -u +%s) + 620 )) -gt "$CLOSE" ]; then stamp "skip $arm: does not end before 04:00Z"; continue; fi
  stamp "start $arm"
  bash experiments/arms/$arm.sh
  stamp "end $arm"
done
stamp "CTL LOO DONE"
