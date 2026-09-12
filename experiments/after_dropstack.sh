#!/usr/bin/env bash
# after_dropstack.sh — wait for `sam_dense`, then run `dropstack_dense` on the freed card. Armed
# 2026-09-12 while sam_dense was on epoch 2 of 12 (534 s/epoch, so ETA ~07:50Z).
#
# UNCONDITIONAL, AND THAT IS DELIBERATE. The two arms are the same axis (generalisation at fixed data and
# fixed architecture) but not the same mechanism, so sam_dense's reading gates NOTHING here: a SAM null
# says the ascent did not help this corner, not that dropping residual branches will not. The only thing
# that would cancel this arm is the card being gone, which the pid branch below handles.
#
# THE LOG IS NAMED FOR THE CHECKPOINT AND THE MARKER FOR THE RUN. The file is `sam_dense_s2026.full.log`
# (basename of --out minus .pt) and the line inside it reads `ARM DONE sam_dense` (run_train's name).
# Greping the checkpoint spelling in the marker waits forever with an idle card behind it.
#
# THE GATE IS AN OR WITH PID LIVENESS, because the marker alone hangs forever on a silent exit-0 death
# while the pid alone fires the next arm the instant a crash frees the card.
#
# NOT SMOKED, AND WHY. `arm_check --smoke` wants the card, and sam_dense is holding 29.3 of 32.6 GB —
# a smoke launched beside it is the GPU-thrash the owner has called out three times. What WAS run is
# `arm_check --only dropstack_dense` with no smoke, which validates every flag against the vocabulary
# without touching the device: "1 arms checked, 0 refused, 0 uncached". That covers the failure a smoke
# is for here (a flag typo costing forty minutes), because the recipe itself is byte-identical to
# `calib_dense_hold`, which has run to completion at this corner.
source "$(dirname "$0")/lib.sh"

LOG="$MODELS/experiments/sam_dense_s2026.full.log"

sam_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*sam_dense*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

while true; do
  grep -aq "ARM DONE sam_dense" "$LOG" 2>/dev/null && break
  sam_alive || break
  sleep 30
done
echo "=== CARD FREE at $(date -u +%H:%MZ) ==="

bash "$(dirname "$0")/arms/dropstack_dense.sh"
