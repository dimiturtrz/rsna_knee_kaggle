#!/usr/bin/env bash
# after_roinative.sh — run the native-resolution ROI arm once the depth24_ff arm frees the card AND the
# roinative mask exists (the cache builder writes it last, knee-vgz5b.10).
set -u

LOG="D:/models/rsna_knee_abnormality/experiments/labelwise_depth24_drop40_long_ff.full.log"
MASK="D:/data/volumetric/mri/processed/rsna_knee_abnormality/mask_336px_20sl_0.1-0.9band_roinative.npy"

running() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*labelwise_depth24_drop40_long_ff*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

echo "waiting for 'ARM DONE labelwise_depth24_drop40_long_ff' in $LOG and $MASK"
until { grep -aq "ARM DONE labelwise_depth24_drop40_long_ff\|ARM FAILED labelwise_depth24_drop40_long_ff" "$LOG" 2>/dev/null || ! running; } && [ -f "$MASK" ]; do
  sleep 60
done
echo "released $(date -u +%H:%MZ)"
bash "$(dirname "$0")/arms/roinative_labelwise_drop40_long.sh"
