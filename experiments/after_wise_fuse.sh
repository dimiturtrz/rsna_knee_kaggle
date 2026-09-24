#!/usr/bin/env bash
# after_wise_fuse.sh — release fuse3_lw12 (knee-vivli) once after_ms_wise.sh has priced the WiSE soups,
# which is the last thing of ours on the card.
set -u
cd "$(dirname "$0")/.."
until grep -aq "=== WISE PRICED ===" logs/after_ms_wise.log 2>/dev/null; do sleep 30; done
echo "=== WISE PRICED SEEN at $(date -u +%H:%MZ) ==="
SMOKE="${TMP:-/tmp}/smoke_fuse3"
.venv/Scripts/python.exe -m kneemri.experiments.arm_check --only fuse3_lw12 --smoke "$SMOKE" --smoke-studies 60 \
  || { echo "=== FUSE3 SMOKE REFUSED at $(date -u +%H:%MZ) ==="; exit 1; }
echo "=== FUSE3 LAUNCH at $(date -u +%H:%MZ) ==="
bash experiments/arms/fuse3_lw12.sh
