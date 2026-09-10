#!/usr/bin/env bash
# after_orthofresh.sh — wait for `ortho_fresh_lowlr`, then run `disagreed_ramp`. Armed 2026-09-10 16:52Z
# while the ortho arm was on its first epoch at 893 img/s.
#
# THERE IS NO GATE BETWEEN THESE TWO and that is deliberate: they sit on different axes. The ortho arm is
# the third point of the rate/initialisation triple; `disagreed_ramp` is the deciding cell on curriculum
# as SCHEDULE (`certainty_ramp` read +0.0077, above the paired floor and convergent with Zhou's +0.0245),
# pre-registered and empty. Neither one's result can license or refuse the other, so this script only
# keeps the card busy in the order the pricing put them.
#
# THE LOG IS NAMED FOR THE CHECKPOINT AND THE MARKER IS NAMED FOR THE RUN. The file is
# `ortho_fresh_lowlr_s2026.full.log` (basename of --out minus .pt) and the line inside reads
# `ARM DONE ortho_fresh_lowlr` (run_train's name). Greping the checkpoint spelling in the marker waits
# forever with an idle card behind it, and `queue_check` reports OK throughout.
#
# AND THE WAIT IS AN OR WITH PID LIVENESS, because the marker alone hangs forever on a silent exit-0
# death while the pid alone fires the next arm the instant a crash frees the card.
source "$(dirname "$0")/lib.sh"

LOG="$MODELS/experiments/ortho_fresh_lowlr_s2026.full.log"

ortho_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*ortho_fresh_lowlr*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

while true; do
  grep -aq "ARM DONE ortho_fresh_lowlr" "$LOG" 2>/dev/null && break
  ortho_alive || break
  sleep 30
done

echo "ortho_fresh_lowlr released the card at $(date -u +%Y-%m-%dT%H:%M:%SZ); starting disagreed_ramp"
bash "$(dirname "$0")/arms/disagreed_ramp.sh"
