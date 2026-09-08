#!/usr/bin/env bash
# Wait for cell D (`ourtrunk44`) to release the card, then run the `final_ws` fold-0 twin (C163 §3).
#
# WHY THIS GATE PATH. `ourtrunk44` is a `foreign_run` arm launched by its own script, which sets
# `LOG="$REPO/logs/ourtrunk44.full.log"` explicitly and echoes `ARM DONE ourtrunk44` into it
# (`experiments/arms/ourtrunk44.sh:47,79`). It is NOT a `run_train` arm, so it does not write under
# `$MODELS/experiments/` — the path below is the producer's own, read off the producer.
set -u

cd /d/personal_projects/rsna_knee_kaggle || exit 1

GATE="logs/ourtrunk44.full.log"
LOG="logs/after_ourtrunk44.log"

echo "waiting for 'ARM DONE ourtrunk44' in $GATE" >> "$LOG"
until grep -q "ARM DONE ourtrunk44" "$GATE" 2>/dev/null; do sleep 30; done
echo "gate cleared at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

bash experiments/arms/final_ws_hold.sh >> "$LOG" 2>&1
echo "final_ws_hold finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
