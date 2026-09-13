#!/usr/bin/env bash
# Wait for severity_drop40_long to release the card, then run labelwise_ext_drop40_long (knee-cgt9m).
#
# THE GATE. `severity_drop40_long` IS a `run_train` arm, so `_mark` fires and the line
# `ARM DONE severity_drop40_long` is written. The FILE is named for the CHECKPOINT
# (`severity_drop40_long_s2026.full.log`) while the MARKER TEXT carries the `run_train` name with no
# seed — two different variables, and grepping the checkpoint spelling in the marker waits forever
# with the card idle behind it. Both spellings below are taken from the arm's `--out` and its
# `run_train` name respectively, not derived from each other.
#
# AND THE LOG IS UNDER $MODELS/experiments, NOT logs/. `lib.sh` re-runs an arm from a snapshot whose
# `tee` writes there. Two watchers have already been armed on a `logs/<checkpoint>.full.log` path
# that does not exist and never will.
#
# THE OR. A marker-only gate hangs forever if the trainer dies without writing one. So the wait ends
# on EITHER the marker appearing or the trainer leaving the process table. The pid is the WINDOWS pid
# read from `card_check`, so liveness is tested with `ps -W` and never with `kill -0` (the MSYS pid is
# a different number and testing it reads a live process as dead).
#
# WHY THIS ARM IS NEXT. Seating (+0.0017 out-of-fold over 140 honest members), architecture (-0.0038)
# and the decision rule (0.0010 spread) are each measured flat. knee-0v8yz: only member quality at
# count 1 has moved this board 0.005+. labelwise_ext_drop40_long is on that axis, and its base is a
# member of the champion six, so a gain needs no reseating to land.
set -u
cd "$(dirname "$0")/.."

PID="${1:?usage: after_labelwise_ext.sh <windows pid of severity_drop40_long>}"
LOG="D:/models/rsna_knee_abnormality/experiments/severity_drop40_long_s2026.full.log"
MARK="ARM DONE severity_drop40_long"

echo "waiting for '$MARK' in $LOG, or pid $PID to exit"
while true; do
  if [ -f "$LOG" ] && grep -q "$MARK" "$LOG"; then echo "marker seen"; break; fi
  if ! ps -W | grep -q "[^0-9]$PID[^0-9]"; then echo "pid $PID gone"; break; fi
  sleep 60
done

sleep 30
echo "launching labelwise_ext_drop40_long at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
bash experiments/arms/labelwise_ext_drop40_long.sh
