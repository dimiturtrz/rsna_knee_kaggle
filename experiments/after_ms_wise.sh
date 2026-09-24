#!/usr/bin/env bash
# after_ms_wise.sh — price the wisepull instrument (backbone pulled toward dinov2-small) once labelwise_ms lands.
set -u
LOG="D:/models/rsna_knee_abnormality/experiments/labelwise_ms_s2026.full.log"
M="D:/models/rsna_knee_abnormality"
while true; do
  grep -aq "ARM DONE labelwise_ms$\|ARM FAILED labelwise_ms$" "$LOG" 2>/dev/null && break
  grep -aqE "ARM (DONE|FAILED) labelwise_ms\b" "$LOG" 2>/dev/null && break
  sleep 30
done
echo "=== LABELWISE_MS RELEASED at $(date -u +%H:%MZ) ==="
cd "$(dirname "$0")/.."
.venv-arm/Scripts/python.exe -m kneemri.inference.holdout_backfill --device cuda \
  --checkpoints "$M/wisepull50_lw24_s2026.pt" "$M/wisepull70_lw24_s2026.pt" "$M/wisepull85_lw24_s2026.pt"
.venv/Scripts/python.exe -m kneemri.evaluation.ledger75 --holdouts "$M" \
  --only labelwise_depth24_drop40_long_s2026 wisepull50_lw24_s2026 wisepull70_lw24_s2026 wisepull85_lw24_s2026 \
         labelwise_ms_ctrl_s2026 labelwise_ms_s2026
echo "=== WISE PRICED at $(date -u +%H:%MZ) ==="
