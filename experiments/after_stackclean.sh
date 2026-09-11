#!/usr/bin/env bash
# after_stackclean.sh -- wait for `distill_stack_max`, then run `distill_stack_clean` unconditionally.
# Armed 2026-09-11 21:46Z while distill_stack_max was on its second epoch.
#
# UNCONDITIONAL ON PURPOSE, unlike after_resnetlong.sh. That watcher gated because 48 epochs on a
# refuted trunk is four times the card spent confirming a defect. Here every branch of
# distill_stack_max's pre-registration leaves this arm worth running: if the stack clears, the teacher
# line is the first peel-back variable; if it lands mid-pack, the same peel-back says whether the
# foreign teacher was the term; and if it lands at or below its distill-only parent, a decorrelated
# member for the honestsix seating is still the cheapest thing the card can make (D373: 45,451 member
# pairs, mean pairwise centred Spearman +0.0144, within-member reliability +0.4965 -- the diversity is
# real and unspent). A gate here would only be able to say "do not measure", which is not a thing any
# branch asks for.
#
# THE LOG IS NAMED FOR THE CHECKPOINT AND THE MARKER FOR THE RUN. File
# `distill_stack_max_s2026.full.log` (basename of --out minus .pt), line inside it `ARM DONE
# distill_stack_max` (run_train's name). Greping the checkpoint spelling in the marker waits forever
# with an idle card behind it, and queue_check reads OK the whole time.
#
# AND THE GATE IS AN OR WITH PID LIVENESS: the marker alone hangs forever on a silent exit-0 death,
# the pid alone fires the next arm the instant a crash frees the card.
source "$(dirname "$0")/lib.sh"

LOG="$MODELS/experiments/distill_stack_max_s2026.full.log"

stack_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*distill_stack_max*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

while true; do
  grep -aq "ARM DONE distill_stack_max" "$LOG" 2>/dev/null && break
  stack_alive || break
  sleep 30
done
echo "=== STACK GATE OPEN at $(date -u +%H:%MZ) ==="

bash "$(dirname "$0")/arms/distill_stack_clean.sh"
