#!/usr/bin/env bash
# after_probes.sh — the window tail after LONG36 PROBES DONE (knee-3tk4b, knee-eduxz).
# Two 12-epoch probes back to back, each ~0.2 h, both ending well inside the 04:00Z close.
set -u
cd "$(dirname "$0")/.."
export UV_NO_SYNC=1
CLOSE=$(date -u -d "2026-09-16T04:00:00Z" +%s)

stamp() { echo "[$(date -u +%H:%MZ)] $*"; }

for spec in probe224_grand_trim:900 probe224_control_noema:900; do
  arm=${spec%%:*}; cost=${spec##*:}
  if [ $(( $(date -u +%s) + cost )) -gt "$CLOSE" ]; then stamp "skip $arm: ${cost}s does not end before 04:00Z"; continue; fi
  stamp "start $arm"
  bash experiments/arms/$arm.sh
  stamp "end $arm"
done
stamp "AFTER PROBES DONE"
