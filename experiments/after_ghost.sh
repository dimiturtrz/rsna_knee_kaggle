#!/usr/bin/env bash
# after_ghost.sh — release labelwise_depth24_distill (knee-bgvmd) once labelwise_depth24_ghost leaves the card.
set -u
cd "$(dirname "$0")/.."
LOG="D:/models/rsna_knee_abnormality/experiments/labelwise_depth24_ghost_s2026.full.log"
until grep -aqE "ARM (DONE|FAILED) labelwise_depth24_ghost" "$LOG" 2>/dev/null; do sleep 30; done
echo "=== GHOST OFF CARD at $(date -u +%H:%MZ) ==="
.venv/Scripts/python.exe -m kneemri.experiments.arm_check --only labelwise_depth24_distill --smoke "${TMP:-/tmp}/smoke_lw24_distill" --smoke-studies 60   || { echo "=== DISTILL SMOKE REFUSED at $(date -u +%H:%MZ) ==="; exit 1; }
echo "=== DISTILL LAUNCH at $(date -u +%H:%MZ) ==="
bash experiments/arms/labelwise_depth24_distill.sh
