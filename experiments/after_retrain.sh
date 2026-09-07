#!/usr/bin/env bash
# The 06:15Z hand-off, decided by a file rather than by whoever is awake.
#
# THE GATE. `retrain_six_distill` is licensed on TWO readings, and only one of them was in hand when this
# was armed. Seat side: `seat_distill_lr4e5` cleared at 0.8632 against a 0.8605 bar, +0.0107 over its
# `lr4e-5_ws` control — done. Board side: `oofsolo` (ref 56065873, submitted 00:50Z) must read >= 0.930.
# If it reads <= 0.928 the +0.0107 is holdout-specific, the teacher's fold structure is the first
# suspect, and multiplying a damped term six times over 8.5 h is the wrong arm to run — `teacher_gen2`
# goes first and grows the term instead.
#
# WHY A FILE AND NOT AN API CALL. The licence is a DECISION taken against a board row, `convergence_check`
# and the bands in the retrain's header — not a threshold a watcher should re-derive at 06:15Z. Writing
# `OK` into `comp/gates/retrain_licence` is how that decision is recorded, and the watcher reads it.
#
# THE DEFAULT IS THE SAFE BRANCH. No file, an unreadable file, or anything other than OK on its first
# line runs `teacher_gen2` — which is mastermind's own fallback, is never wasted work on either branch
# (gen-2 is the second lever regardless), and leaves the retrain to be armed deliberately afterwards.
# A watcher that guessed the other way would spend 8.5 h of card on an unlicensed arm while nobody
# was watching.
set -u
cd "$(dirname "$0")/.."

MARKER="ARM DONE distill_v3base_oof"
ARM_LOG="D:/models/rsna_knee_abnormality/experiments/distill_v3base_oof_s2026.full.log"
LICENCE="comp/gates/retrain_licence"

until grep -q "$MARKER" "$ARM_LOG" 2>/dev/null; do sleep 60; done
echo "$MARKER seen — reading the retrain licence $(date -u +%FT%TZ)"

if [ "$(head -1 "$LICENCE" 2>/dev/null)" = "OK" ]; then
  echo "LICENCE OK — six-seat retrain $(date -u +%FT%TZ)"
  bash experiments/arms/retrain_six_distill.sh
  echo "QUEUE retrain_six_distill exited $? $(date -u +%FT%TZ)"
  bash experiments/arms/teacher_gen2.sh
  echo "QUEUE teacher_gen2 exited $? $(date -u +%FT%TZ)"
else
  echo "LICENCE ABSENT — gen-2 teacher first, retrain armed deliberately after $(date -u +%FT%TZ)"
  bash experiments/arms/teacher_gen2.sh
  echo "QUEUE teacher_gen2 exited $? $(date -u +%FT%TZ)"
fi
