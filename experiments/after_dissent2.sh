#!/usr/bin/env bash
# Replaces after_dissent.sh: the same wait, with oof_teacher inserted between the sweep and mix30.
#
# WHY A SECOND FILE RATHER THAN AN EDIT. bash reads a script by byte offset as it executes, so editing
# the file a live watcher is sitting inside changes what it runs next. The watcher had not reached the
# card yet (it was still in the wait loop), so replacing the process is free and editing under it is not.
#
# ORDER, per mastermind 20:23Z. The fine sweep is on a clock (a clearing band has to be pushed for the
# 22:30Z census). `oof_teacher` goes AHEAD of mix30 because mix30's own pre-registration predicts a null
# — it is a closing instrument — while oof_teacher is the only untried distillation form whose teacher
# reaches every cell of the label defect. mix30 slipping to tomorrow is not a loss.
set -u
cd "$(dirname "$0")/.."

[ $# -ge 1 ] || { echo "usage: $(basename "$0") <windows-pid to wait on>" >&2; exit 2; }

until ! tasklist //FI "PID eq ${1}" 2>/dev/null | grep -q "${1}"; do sleep 60; done
echo "ARM DONE foreign_dissent_s2026 — starting the queue $(date -u +%FT%TZ)"

bash experiments/arms/foreign_decode_fine.sh
echo "QUEUE fine sweep exited $? $(date -u +%FT%TZ)"

bash experiments/arms/oof_teacher.sh
echo "QUEUE oof_teacher exited $? $(date -u +%FT%TZ)"

bash experiments/arms/foreign_distill_mix30.sh
echo "QUEUE mix30 exited $? $(date -u +%FT%TZ)"
