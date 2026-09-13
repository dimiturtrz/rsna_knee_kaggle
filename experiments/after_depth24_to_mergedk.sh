#!/usr/bin/env bash
# Wait for labelwise_depth24_drop40_long to release the card, then run mergedk_labelwise_drop40_long
# (knee-dl65h).
#
# THE GATE. `labelwise_depth24_drop40_long` IS a `run_train` arm, so `_mark` fires. The FILE is named
# for the CHECKPOINT (`labelwise_depth24_drop40_long_s2026.full.log`) while the MARKER TEXT carries the
# `run_train` name with NO seed — two different variables. Both spellings below come from the arm's
# `--out` and its `run_train` name respectively, not derived from each other. The log path was `ls`-ed
# before this watcher was armed.
#
# AND THE LOG IS UNDER $MODELS/experiments, NOT logs/.
#
# THE OR. A marker-only gate hangs forever if the trainer dies without writing one, and a pid-only gate
# fires the next arm the instant a crash frees the card. Both, with OR. `ps -W` because the MSYS pid is
# a different number from the Windows one.
#
# WHY THIS ARM IS NEXT, UNCONDITIONALLY. It is matched against `mergedk_drop40_long`, not against
# anything the depth24 arm produces, so neither branch of that readout changes whether this cell is
# worth filling. Queuing it blind costs nothing and keeps the card busy across the boundary.
set -u
cd "$(dirname "$0")/.."

PID="${1:?usage: after_depth24_to_mergedk.sh <windows pid of labelwise_depth24_drop40_long>}"
LOG="D:/models/rsna_knee_abnormality/experiments/labelwise_depth24_drop40_long_s2026.full.log"
MARK="ARM DONE labelwise_depth24_drop40_long"

echo "waiting for '$MARK' in $LOG, or pid $PID to exit"
while true; do
  if [ -f "$LOG" ] && grep -q "$MARK" "$LOG"; then echo "marker seen"; break; fi
  if ! ps -W | grep -q "[^0-9]$PID[^0-9]"; then echo "pid $PID gone"; break; fi
  sleep 60
done

sleep 30
echo "launching mergedk_labelwise_drop40_long at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
bash experiments/arms/mergedk_labelwise_drop40_long.sh
