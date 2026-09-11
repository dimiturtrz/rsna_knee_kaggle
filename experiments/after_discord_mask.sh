#!/usr/bin/env bash
# after_discord_mask.sh — wait for `discord_mask`, then run `v2_corrected`. Armed 2026-09-11 14:08Z with
# discord_mask at epoch 1.
#
# UNGATED on discord_mask's number, deliberately, and for a different reason than after_discord.sh's.
# `v2_corrected` is not the third masking arm — it is the CORRECTION arm, the other branch of D333's
# ladder, on a selector (2,315 cells, 75% wrong against two independent references) that neither masking
# arm used. Its three-way pre-registration is written against band_span 0.8565, not against discord_mask,
# so nothing discord_mask can return changes what v2_corrected measures or whether it is worth running.
# Gating it would only idle the card.
#
# THE LOG IS NAMED FOR THE CHECKPOINT AND THE MARKER FOR THE RUN: file `discord_mask_s2026.full.log`,
# line `ARM DONE discord_mask`. The gate is an OR with pid liveness — the marker alone hangs forever on a
# silent exit-0 death, the pid alone fires the next arm the instant a crash frees the card.
source "$(dirname "$0")/lib.sh"

LOG="$MODELS/experiments/discord_mask_s2026.full.log"

prev_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*discord_mask*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

while true; do
  grep -aq "ARM DONE discord_mask" "$LOG" 2>/dev/null && break
  prev_alive || break
  sleep 30
done
echo "=== CARD FREE at $(date -u +%H:%MZ) - launching v2_corrected ==="
bash "$(dirname "$0")/arms/v2_corrected.sh"
