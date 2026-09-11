#!/usr/bin/env bash
# after_discord.sh — wait for `confident_span`, then run `discord_mask`. Armed 2026-09-11 13:38Z with
# confident_span at step ~60/291.
#
# UNGATED, deliberately. `discord_mask` is not a follow-on to confident_span's result — it is the SECOND
# SELECTOR on the same axis, and the two are disjoint (188 cells overlap, 2.55% of discordant). Gating it
# on confident_span's number would throw away the one reading that needs both: same recipe, same control
# (band_span 0.8565), two different ways of deciding which cells to stop trusting. Anticorrelation between
# them is the informative outcome and it is unreadable if only one runs.
#
# THE LOG IS NAMED FOR THE CHECKPOINT AND THE MARKER FOR THE RUN: file `confident_span_s2026.full.log`,
# line `ARM DONE confident_span`. And the gate is an OR with pid liveness, because the marker alone hangs
# forever on a silent exit-0 death while the pid alone fires the next arm the instant a crash frees the card.
source "$(dirname "$0")/lib.sh"

LOG="$MODELS/experiments/confident_span_s2026.full.log"

prev_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*confident_span*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

while true; do
  grep -aq "ARM DONE confident_span" "$LOG" 2>/dev/null && break
  prev_alive || break
  sleep 30
done
echo "=== CARD FREE at $(date -u +%H:%MZ) - launching discord_mask ==="
bash "$(dirname "$0")/arms/discord_mask.sh"
