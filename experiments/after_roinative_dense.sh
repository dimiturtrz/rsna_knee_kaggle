#!/usr/bin/env bash
# after_roinative_dense.sh — run labelwise_dense_drop40_long_ff once the roinative arm frees the card (knee-sexix).
set -u

LOG="D:/models/rsna_knee_abnormality/experiments/roinative_labelwise_drop40_long_s2026.full.log"

running() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*roinative_labelwise_drop40_long*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

echo "waiting for 'ARM DONE roinative_labelwise_drop40_long' in $LOG"
until grep -aq "ARM DONE roinative_labelwise_drop40_long\|ARM FAILED roinative_labelwise_drop40_long" "$LOG" 2>/dev/null || ! running; do
  sleep 60
done
echo "released $(date -u +%H:%MZ)"
bash "$(dirname "$0")/arms/labelwise_dense_drop40_long_ff.sh"
