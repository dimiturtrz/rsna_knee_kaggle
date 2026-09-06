#!/usr/bin/env bash
# Waits for foreign_distill_mix30 to finish, then runs knee-wko44's arm A (labels_v4hybrid).
#
# The wait is on the MARKER rather than on a pid, because mix30 has not started yet — it is third in
# after_dissent2.sh's queue, behind oof_teacher — so there is no pid to hand this script.
#
# THE MARKER IS SPELLED WITHOUT THE SEED, and that is the whole reason this file was rewritten once.
# `_mark "$name" "ARM DONE $name"` (experiments/lib.sh:642) writes the run_train NAME, while
# `_arm_log` (:685) names the FILE for the checkpoint basename via ARM_LOG_NAME (:580). So an arm
# whose `--out` carries `_s2026` writes `foreign_distill_mix30_s2026.full.log` containing the line
# `ARM DONE foreign_distill_mix30` — the file has the seed and the marker does not. The first version
# of this watcher grepped the checkpoint spelling in the marker and would have waited forever with the
# card idle behind it; queue_check reported OK throughout, because a live process gating an owed arm
# is all it can see. Proof of the format, from a marker already on disk:
# `logs/after_dissent2.log:139: ARM DONE foreign_decode_fine 2026-09-06T20:56:18Z`.
#
# `_mark` tees to BOTH the arm log and stdout, and the chain's stdout is after_dissent2.log — which is
# where every marker so far has provably landed, so that is the file watched here.
#
# Arm A is queued LAST deliberately. mix30's own pre-registration predicts a null (it is a closing
# instrument for the distillation-mixture axis) while arm A is the label-table axis's only card-priced
# test, so arm A must not be the thing that slips if the card is taken. But mix30 is already armed and
# a live watcher is replaced by killing it, never by editing it mid-run — so the cheap correct move is
# to queue behind it rather than to rebuild the chain.
set -u
cd "$(dirname "$0")/.."

MARKER="ARM DONE foreign_distill_mix30"
LOG="logs/after_dissent2.log"
ARM_LOG="logs/foreign_distill_mix30_s2026.full.log"

until grep -q "$MARKER" "$LOG" "$ARM_LOG" 2>/dev/null; do sleep 60; done
echo "$MARKER seen — starting labels_v4hybrid $(date -u +%FT%TZ)"

bash experiments/arms/labels_v4hybrid.sh
echo "QUEUE labels_v4hybrid exited $? $(date -u +%FT%TZ)"
