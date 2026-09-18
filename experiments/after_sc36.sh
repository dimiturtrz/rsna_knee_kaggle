#!/usr/bin/env bash
# Waits for the 36-slice corpus decode to exit, then gates and launches slicecover36 (knee-jb0ci).
# The decode is not an arm and writes no ARM DONE marker, so the gate is the ARTEFACT (the npy is
# fully written and the log says so) OR-ed with pid liveness, the foreign_run rule.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
LOG="$REPO/logs/build_36sl.log"
CACHE="D:/data/volumetric/mri/processed/rsna_knee_abnormality/images_336px_36sl_0.1-0.9band.npy"

while true; do
  alive=0
  for pid in 17232 38636; do
    ps -W 2>/dev/null | grep -q "[^0-9]${pid}[^0-9]" && alive=1
  done
  [ "$alive" = "0" ] && break
  sleep 120
done

echo "DECODE EXITED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
tail -3 "$LOG"

if ! grep -aqi "done\|complete\|wrote" "$LOG" && [ ! -s "$CACHE" ]; then
  echo "REFUSED $(date -u +%Y-%m-%dT%H:%M:%SZ) - decode left no usable cache, card left free"
  exit 1
fi

SMOKE="D:/models/rsna_knee_abnormality/smoke_sc36_$(date -u +%H%M%S)"
if UV_NO_SYNC=1 "$REPO/.venv/Scripts/python.exe" -m kneemri.experiments.reading_check --only slicecover36 \
   && UV_NO_SYNC=1 "$REPO/.venv/Scripts/python.exe" -m kneemri.experiments.card_check \
   && UV_NO_SYNC=1 "$REPO/.venv/Scripts/python.exe" -m kneemri.experiments.arm_check --only slicecover36 --smoke "$SMOKE" --smoke-studies 60; then
  echo "GATES OK $(date -u +%Y-%m-%dT%H:%M:%SZ) - launching slicecover36"
  bash "$HERE/arms/slicecover36.sh"
else
  echo "REFUSED slicecover36 $(date -u +%Y-%m-%dT%H:%M:%SZ) - card left free"
  exit 1
fi
