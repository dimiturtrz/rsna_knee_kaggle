#!/usr/bin/env bash
# after_distill.sh — re-run labelwise_depth24_ghost (knee-mum8w, OOM at epoch 5 on 2026-09-25) once labelwise_depth24_distill leaves the card.
set -u
cd "$(dirname "$0")/.."
LOG="D:/models/rsna_knee_abnormality/experiments/labelwise_depth24_distill_s2026.full.log"
until grep -aqE "ARM (DONE|FAILED) labelwise_depth24_distill" "$LOG" 2>/dev/null; do sleep 30; done
echo "=== DISTILL OFF CARD at $(date -u +%H:%MZ) ==="
echo "=== GHOST RELAUNCH at $(date -u +%H:%MZ) ==="
ALLOW_OVERWRITE=1 bash experiments/arms/labelwise_depth24_ghost.sh --resume
