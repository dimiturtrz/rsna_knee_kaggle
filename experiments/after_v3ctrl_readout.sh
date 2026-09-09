#!/usr/bin/env bash
# Wait for v3small_control to release the card, then run the foreign-decode readout: pilot, then pool.
#
# THE GATE IS THE MARKER *OR* THE pid. `v3small_control` is a `run_train` arm, so `_mark` fires — but the
# FILE is named for the CHECKPOINT (`v3small_control_s2026.full.log`, under $MODELS/experiments/, not
# logs/) while the marker TEXT carries the `run_train` name WITHOUT the seed. Verified against the arm:
# `run_train v3small_control` at experiments/arms/v3small_control.sh:39, `--out .../v3small_control_s2026.pt`
# at :42. The pid arm covers the fourth failure: a death writes no marker at all.
#
# THE pid IS THE WINDOWS pid from `card_check`, so liveness is `ps -W`, never `kill -0`.
#
# THE PILOT GATES THE POOL. `slot_cost` is a plumbing self-check that must read ~0.0000; anything else
# means the shifted slab is broken and `decode_cost` is not about pixels. The pool run does not start
# unless both pilot members read inside +/-0.005 — a broken slab would otherwise produce eleven
# plausible numbers that mean nothing.
set -u
cd "$(dirname "$0")/.."

PID="${1:?usage: after_v3ctrl_readout.sh <windows pid of v3small_control>}"
LOG="D:/models/rsna_knee_abnormality/experiments/v3small_control_s2026.full.log"
TOL=0.005

echo "waiting for 'ARM DONE v3small_control' in $LOG / pid $PID at $(date -u +%FT%TZ)"
until grep -q "ARM DONE v3small_control" "$LOG" 2>/dev/null || ! ps -W | grep -q "[^0-9]$PID[^0-9]"; do
  sleep 60
done
echo "card released at $(date -u +%FT%TZ); marker present: $(grep -qs 'ARM DONE v3small_control' "$LOG" && echo yes || echo no)"

bash experiments/arms/foreign_drop_pilot.sh 2>&1 | tee logs/foreign_drop_pilot.log
echo "PILOT foreign_drop_pilot exited ${PIPESTATUS[0]} $(date -u +%FT%TZ)"

WORST=$(grep -o "slots [+-][0-9.]*" logs/foreign_drop_pilot.log | awk '{v=$2<0?-$2:$2; if(v>m) m=v} END {printf "%.4f", m+0}')
COUNT=$(grep -c "slots [+-][0-9.]*" logs/foreign_drop_pilot.log)
echo "pilot: $COUNT readings, worst |slot_cost| $WORST against tolerance $TOL"

if [ "$COUNT" -lt 2 ]; then
  echo "REFUSED: pilot produced $COUNT readings, expected 2 - the pool run does not start"
  exit 1
fi
if awk -v w="$WORST" -v t="$TOL" 'BEGIN{exit !(w>t)}'; then
  echo "REFUSED: slot_cost $WORST exceeds $TOL - the shifted path is broken and decode_cost is not about pixels"
  exit 1
fi

bash experiments/arms/foreign_drop_pool.sh 2>&1 | tee logs/foreign_drop_pool.log
echo "QUEUE foreign_drop_pool exited ${PIPESTATUS[0]} $(date -u +%FT%TZ)"
