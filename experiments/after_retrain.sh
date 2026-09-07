#!/usr/bin/env bash
# Chains the six-seat retrain behind the oof5 student, and compose behind THAT (mastermind 03:03Z:
# the retrain is the lever to first place, so it takes the card before the 1.2 h composition question).
#
# THE MARKER IS SPELLED WITHOUT THE SEED AND THE LOG IS SPELLED WITH IT. `_mark` writes the `run_train`
# name (`ARM DONE seat_oof5_distill`) while `_arm_log` names the FILE for the checkpoint basename
# (`seat_oof5_distill_s2026.full.log`). Grepping the checkpoint spelling in the marker text waits
# forever with the card idle behind it, and `queue_check` reports OK the whole time.
set -u
cd "$(dirname "$0")/.."

MARKER="ARM DONE seat_oof5_distill"
ARM_LOG="D:/models/rsna_knee_abnormality/experiments/seat_oof5_distill_s2026.full.log"

until grep -q "$MARKER" "$ARM_LOG" 2>/dev/null; do sleep 60; done
echo "$MARKER seen $(date -u +%FT%TZ)"

bash experiments/arms/retrain_six_distill.sh
echo "QUEUE retrain_six_distill exited $? $(date -u +%FT%TZ)"

bash experiments/arms/compose_v4b_clean.sh
echo "QUEUE compose_v4b_clean exited $? $(date -u +%FT%TZ)"
