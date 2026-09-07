#!/usr/bin/env bash
# Wait for `effv2l_f0` to release the card, then run `clean_calib` (D29's knob on D30's clean parent).
#
# WHY THE MARKER AND NOT A PID. `after_effv2l.sh:76` writes `ARM DONE effv2l_f0` into
# `logs/raptor_ft_effv2l_f0.full.log` when its `wait` returns; the training pid does not exist yet at
# arming time, so the string is the only signal available before the fact. The marker text is the
# `--tag`/`run_train` name and the FILE is named for the checkpoint — two different spellings, and this
# greps the one an existing marker line actually carries (`CLAUDE.md`, knee-ket1c).
#
# It does NOT preempt: effv2l has the card until it says so.
set -u

cd /d/personal_projects/rsna_knee_kaggle || exit 1

GATE="logs/raptor_ft_effv2l_f0.full.log"
LOG="logs/after_clean_calib.log"

echo "waiting for 'ARM DONE effv2l_f0' in $GATE" >> "$LOG"
until grep -q "ARM DONE effv2l_f0" "$GATE" 2>/dev/null; do sleep 30; done
echo "gate cleared at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

bash experiments/arms/clean_calib.sh >> "$LOG" 2>&1
echo "clean_calib watcher finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
