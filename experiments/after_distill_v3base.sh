#!/usr/bin/env bash
# Queues the far-family distillation arm behind `compose_v4b_oof`, per mastermind 00:24Z.
#
# WHY A SIXTH CHAIN FILE AND NOT AN EDIT TO after_compose.sh. That shell is live and blocked on its own
# grep loop; bash reads a script by byte offset as it executes, so appending to it rewrites bytes the
# live shell is about to read. Same reason after_compose.sh exists rather than an edit to after_seat.sh.
#
# MARKER AND LOG ARE SPELLED DIFFERENTLY AND BOTH ARE GREPPED OFF THE ARM, NEVER DERIVED. `_arm_log`
# names the FILE from the basename of --out (so the seed is in it); `_mark` writes the run_train NAME
# (so the seed is not). compose_v4b_oof.sh: `run_train compose_v4b_oof` with
# `--out $MODELS/compose_v4b_oof_s2026.pt`.
set -u
cd "$(dirname "$0")/.."

MARKER="ARM DONE compose_v4b_oof"
ARM_LOG="D:/models/rsna_knee_abnormality/experiments/compose_v4b_oof_s2026.full.log"

until grep -q "$MARKER" "$ARM_LOG" 2>/dev/null; do sleep 60; done
echo "$MARKER seen — starting the far-family distillation arm $(date -u +%FT%TZ)"

bash experiments/arms/distill_v3base_oof.sh
echo "QUEUE distill_v3base_oof exited $? $(date -u +%FT%TZ)"
