#!/usr/bin/env bash
# Run the readout immediately: pilot, then pool, with the pilot gating the pool on its plumbing check.
# Same gate as after_v3ctrl_readout.sh, without the wait — the card is already free.
set -u
cd "$(dirname "$0")/.."
TOL=0.005

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
