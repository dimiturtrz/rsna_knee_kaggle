#!/usr/bin/env bash
# after_dropdrop.sh — wait for `labelwise_drop_drop40_long` (knee-fnhbs), then run
# `clean_calib_drop40_long` (knee-rpugl). Armed 2026-09-13 11:40Z with the fourth base on epoch 5 of 24.
#
# ARMED BESIDE THE RUNNING ARM, NOT AFTER IT — the fix that worked on `after_densedrop.sh`, which
# released within seconds of its marker. The card sat idle 09:11Z to 10:02Z the one time a successor was
# paced off an ESTIMATED landing instead of an armed watcher.
#
# NO GATE ON THE FOURTH BASE'S NUMBER, DELIBERATELY, and the reason is sharper than last time: the two
# arms are now on OPPOSITE SIDES of a distinction that was measured after `labelwise_drop_drop40_long`
# was armed. That arm asks whether upgraded bases keep ADDING seats. This one asks whether an upgraded
# base is worth more as a SWAP. Measured 2026-09-13, ADD and SWAP disagree — chainfour_up adds +0.0007
# and swaps +0.0012, blendhold_up adds +0.0000 and swaps +0.0008 — so a flat ADD result cannot speak to
# the SWAP question and must not block it.
#
# THE LOG IS NAMED FOR THE CHECKPOINT AND THE MARKER FOR THE RUN NAME — two separate variables
# (lib.sh:580 vs :642), written here from their two separate sources rather than derived from one.
#
# AND THE GATE IS AN OR WITH PID LIVENESS, because the marker alone hangs forever on a silent exit-0
# death while the pid alone fires the next arm the instant a crash frees the card.
source "$(dirname "$0")/lib.sh"

LOG="$MODELS/experiments/labelwise_drop_drop40_long_s2026.full.log"

fourth_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*labelwise_drop_drop40_long*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

while true; do
  grep -aq "ARM DONE labelwise_drop_drop40_long" "$LOG" 2>/dev/null && break
  fourth_alive || break
  sleep 30
done
echo "=== FOURTH BASE RELEASED at $(date -u +%H:%MZ) ==="

# THE INSTRUMENT RUNS FIRST, and the ordering is a price decision rather than a preference.
# `clean_calib_drop40_long` (knee-rpugl) costs ~2.4 h and improves a PROXY pool that has no board anchor
# — `cleanseat_v1` is not the honest core six, so nothing on the board prices that pool.
# `stack_increments_drop40_long` (knee-p29bb) costs ~1.6 h and is the gating instrument of knee-289qc:
# it reads whether the operator pair reaches the CHAMPION's own recipe family, where all six seats'
# fold-0 twins peak at epoch 6-9 of 12 and decline. Cheaper, and it decides twelve hours of card.
bash "$(dirname "$0")/arms/stack_increments_drop40_long.sh"
bash "$(dirname "$0")/arms/clean_calib_drop40_long.sh"
