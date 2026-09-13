#!/usr/bin/env bash
# Wait for labelwise_depth24_drop40_long to release the card, then run labelwise_ext_drop40_long_ff
# (knee-sexix).
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
# WHY THIS ARM IS NEXT. The champion's six seats are all --fold -1 members that saw every study; our
# best honest solo saw ~3500 of ~4400. A full-corpus refit of the winning recipe is the only operation
# that puts a better member into a seat the champion already holds. Independent of the depth24 readout.
set -u
cd "$(dirname "$0")/.."

PID="${1:?usage: after_depth24_to_ff.sh <windows pid of labelwise_depth24_drop40_long>}"
LOG="D:/models/rsna_knee_abnormality/experiments/labelwise_depth24_drop40_long_s2026.full.log"
MARK="ARM DONE labelwise_depth24_drop40_long"

echo "waiting for '$MARK' in $LOG, or pid $PID to exit"
while true; do
  if [ -f "$LOG" ] && grep -q "$MARK" "$LOG"; then echo "marker seen"; break; fi
  if ! ps -W | grep -q "[^0-9]$PID[^0-9]"; then echo "pid $PID gone"; break; fi
  sleep 60
done

sleep 30
echo "launching labelwise_ext_drop40_long_ff at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
bash experiments/arms/labelwise_ext_drop40_long_ff.sh
