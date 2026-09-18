#!/usr/bin/env bash
# after_auxpair.sh — wait for `headlr_ship_ff`, then run knee-evdfd's pair, control first.
#
# WHY CHAINED AND NOT PICKED LATER. `headlr_ship_ff` fits `--fold -1`, so it holds no study back and no
# ledger row can price it — its reading is a board slot, hours after it finishes. It therefore cannot
# choose its own successor in either branch, and waiting on it buys nothing but an idle card.
#
# WHY THIS PAIR. knee-evdfd, priced at ~2.5 h with P(clears +0.0040) ~ 0.25. It is bought as an
# INSTRUMENT: `band_dense_drop40_long` is the matched converged control that re-prices all twelve
# --*-weight arms, whose numbers are currently facts about a 12-epoch wall rather than about objectives.
# The control runs FIRST because it is reusable whatever happens to the treatment.
#
# AND THE RULER STILL WORKS HERE. 2026-09-18's ruler ceiling closes offline ranking ABOVE offline75
# 0.904; this pair sits near 0.887 at the 224/20-slice corner, where the ruler reads resid sd 0.0021
# against a 0.030 span.
#
# THE LOG IS NAMED FOR THE CHECKPOINT AND THE MARKER FOR THE RUN, and the wait is an OR with pid
# liveness so a silent exit-0 death does not hang it forever.
source "$(dirname "$0")/lib.sh"

LOG="$MODELS/experiments/headlr_ship_ff.full.log"

arm_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*headlr_ship_ff*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

echo "waiting on headlr_ship_ff / $LOG at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
while true; do
  grep -aq "ARM DONE headlr_ship_ff" "$LOG" 2>/dev/null && break
  arm_alive || break
  sleep 30
done
echo "card released at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

cd "$REPO" || exit 1
for arm in band_dense_drop40_long severity_drop40_long; do
  SMOKE="$MODELS/smoke_${arm}_$(date -u +%H%M%S)"
  if UV_NO_SYNC=1 .venv/Scripts/python.exe -m kneemri.experiments.reading_check --only "$arm" \
     && UV_NO_SYNC=1 .venv/Scripts/python.exe -m kneemri.experiments.arm_check \
          --only "$arm" --smoke "$SMOKE" --smoke-studies 60; then
    echo "SMOKE OK $(date -u +%Y-%m-%dT%H:%M:%SZ) - launching $arm"
    bash "$(dirname "$0")/arms/${arm}.sh"
  else
    echo "REFUSED $arm $(date -u +%Y-%m-%dT%H:%M:%SZ) - card left free, chain stops"
    break
  fi
done
