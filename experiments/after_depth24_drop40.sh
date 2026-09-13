#!/usr/bin/env bash
# Wait for labelwise_ext_drop40_long to release the card, then run labelwise_depth24_drop40_long
# (knee-9bnkc).
#
# THE GATE. `labelwise_ext_drop40_long` IS a `run_train` arm, so `_mark` fires. The FILE is named for
# the CHECKPOINT (`labelwise_ext_drop40_long_s2026.full.log`) while the MARKER TEXT carries the
# `run_train` name with NO seed — two different variables, and grepping the checkpoint spelling in the
# marker waits forever with the card idle behind it. Both spellings below come from the arm's `--out`
# and its `run_train` name respectively, not derived from each other.
#
# AND THE LOG IS UNDER $MODELS/experiments, NOT logs/.
#
# THE OR. A marker-only gate hangs forever if the trainer dies without writing one, and a pid-only gate
# fires the next arm the instant a crash frees the card. Both, with OR. `ps -W` because the MSYS pid is
# a different number from the Windows one.
#
# WHY THIS ARM IS NEXT, UNCONDITIONALLY. Both branches of knee-cgt9m's readout want it — a clear makes
# the family's best base the obvious next target, a flat needs a matched second base before the
# operator-transfer question may close. See the arm header.
set -u
cd "$(dirname "$0")/.."

PID="${1:?usage: after_depth24_drop40.sh <windows pid of labelwise_ext_drop40_long>}"
LOG="D:/models/rsna_knee_abnormality/experiments/labelwise_ext_drop40_long_s2026.full.log"
MARK="ARM DONE labelwise_ext_drop40_long"

echo "waiting for '$MARK' in $LOG, or pid $PID to exit"
while true; do
  if [ -f "$LOG" ] && grep -q "$MARK" "$LOG"; then echo "marker seen"; break; fi
  if ! ps -W | grep -q "[^0-9]$PID[^0-9]"; then echo "pid $PID gone"; break; fi
  sleep 60
done

sleep 30
echo "launching labelwise_depth24_drop40_long at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
bash experiments/arms/labelwise_depth24_drop40_long.sh
