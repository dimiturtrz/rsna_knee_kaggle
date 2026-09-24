#!/usr/bin/env bash
# after_msctrl.sh — release labelwise_ms (knee-5jfix) when `labelwise_ms_ctrl` lands AND the multiscale
# cache build has exited. The smoke runs here, not earlier: before this point the control holds the card.
#
# The log is named for the CHECKPOINT (`labelwise_ms_ctrl_s2026.full.log`) while the MARKER TEXT carries the
# `run_train` name (`ARM DONE labelwise_ms_ctrl`).
set -u

LOG="D:/models/rsna_knee_abnormality/experiments/labelwise_ms_ctrl_s2026.full.log"
SMOKE="${TMP:-/tmp}/smoke_labelwise_ms"

count() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '$1' }).Count" \
    2>/dev/null | tr -d '\r'
}

while true; do
  grep -aq "ARM DONE labelwise_ms_ctrl" "$LOG" 2>/dev/null && break
  grep -aq "ARM FAILED labelwise_ms_ctrl" "$LOG" 2>/dev/null && break
  [ "$(count '*labelwise_ms_ctrl*')" = "0" ] && break
  sleep 30
done
echo "=== MSCTRL RELEASED at $(date -u +%H:%MZ) ==="

while [ "$(count '*--scheme multiscale*')" != "0" ]; do sleep 30; done
echo "=== MULTISCALE CACHE BUILD GONE at $(date -u +%H:%MZ) ==="

cd "$(dirname "$0")/.."
.venv/Scripts/python.exe -m kneemri.experiments.arm_check --only labelwise_ms --smoke "$SMOKE" --smoke-studies 60 \
  || { echo "=== LABELWISE_MS SMOKE REFUSED at $(date -u +%H:%MZ) ==="; exit 1; }
echo "=== LABELWISE_MS LAUNCH at $(date -u +%H:%MZ) ==="
bash experiments/arms/labelwise_ms.sh
