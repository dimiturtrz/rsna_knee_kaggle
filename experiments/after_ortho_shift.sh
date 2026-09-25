#!/usr/bin/env bash
# after_ortho_shift.sh — release shift4_lw12 (knee-ypu31) once the OrthoDiffusion Stage II fine-tune
# (knee-mxz08) leaves the card: its own DONE line OR its Windows pid gone, so a crash frees it too.
set -u
cd "$(dirname "$0")/.."
LOG="D:/models/rsna_knee_abnormality/experiments/orthodiff_ft_s2026.full.log"
PID=50392
until grep -aq "ORTHO FINETUNE DONE" "$LOG" 2>/dev/null || ! ps -W | grep -q "[^0-9]$PID[^0-9]"; do sleep 30; done
echo "=== ORTHO OFF CARD at $(date -u +%H:%MZ) ==="
sleep 60
SMOKE="${TMP:-/tmp}/smoke_shift4"
.venv/Scripts/python.exe -m kneemri.experiments.arm_check --only shift4_lw12 --smoke "$SMOKE" --smoke-studies 60 \
  || { echo "=== SHIFT4 SMOKE REFUSED at $(date -u +%H:%MZ) ==="; exit 1; }
echo "=== SHIFT4 LAUNCH at $(date -u +%H:%MZ) ==="
bash experiments/arms/shift4_lw12.sh
