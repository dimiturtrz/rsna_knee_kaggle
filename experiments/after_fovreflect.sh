#!/usr/bin/env bash
# Wait for fovjitter160 to release the card, then run its padding disambiguation.
#
# THE GATE IS THE CHECKPOINT-SPELLED LOG, AND THE MARKER TEXT IS THE run_train NAME. `_arm_log` names
# the file from `basename "$out" .pt` (`fovjitter160_s2026.full.log`) while `_mark` writes the
# `run_train` name (`ARM DONE fovjitter160`) — two different variables, and only one of them carries
# the seed. That file lives under `$MODELS/experiments/`, never under `logs/`: a `run_train` arm is
# re-run from a snapshot and the snapshot's `tee` writes there.
#
# BOTH MARKERS END THE WAIT. `ARM FAILED` is a card release exactly like `ARM DONE` — the previous
# incarnation of this arm died at epoch 7 of 12 on an `nvidia-smi` timeout (fixed in `card_memory`),
# and a watcher that only greps DONE would have held the card idle behind a corpse.
set -u
cd "$(dirname "$0")/.."

LOG="D:/models/rsna_knee_abnormality/experiments/fovjitter160_s2026.full.log"

echo "waiting on fovjitter160 via $LOG at $(date -u +%FT%TZ)"
until grep -qE "ARM (DONE|FAILED) fovjitter160" "$LOG" 2>/dev/null; do sleep 60; done
echo "card released at $(date -u +%FT%TZ): $(grep -E "ARM (DONE|FAILED) fovjitter160" "$LOG" | tail -1)"

bash experiments/arms/fovjitter160_reflect.sh
echo "QUEUE fovjitter160_reflect exited $? $(date -u +%FT%TZ)"
