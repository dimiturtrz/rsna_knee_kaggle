#!/usr/bin/env bash
# Chains compose_v4b_clean behind the oof5 student so the card does not idle at ~06:00Z.
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

bash experiments/arms/compose_v4b_clean.sh
echo "QUEUE compose_v4b_clean exited $? $(date -u +%FT%TZ)"
