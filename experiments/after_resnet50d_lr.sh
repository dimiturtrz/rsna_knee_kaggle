#!/usr/bin/env bash
# Wait for `resnet50d_lr` to release the card, then run cell D (`ourtrunk44`).
#
# THIS FILE REPLACES `after_trunkdraws.sh`, WHICH WAS DEAD AND WOULD HAVE DOUBLE-RUN AN ARM. That
# watcher gated on `ARM DONE deep32_batch6`, and `deep32_batch6` never produced one: it died on a
# missing 95 GB cache (`images_336px_32sl_0.1-0.9band.npy`) and wrote `ARM FAILED` instead. So it would
# have waited forever with the card idle behind it — and its payload was `resnet50d_lr.sh`, which had
# already been launched by hand, so the one thing worse than it hanging was it firing. Killed 02:20Z.
#
# WHY THIS GATE PATH AND THIS MARKER TEXT. A `run_train` arm re-runs from a snapshot whose `tee` writes
# `$MODELS/experiments/<checkpoint>.full.log` — the FILE carries the checkpoint spelling
# (`resnet50d_lr_s2026`) while the MARKER carries the `run_train` name (`resnet50d_lr`). Two different
# variables, and `queue_check` cannot see a watcher blocked on the wrong one: a live process gating an
# owed arm is all it reads.
set -u

cd /d/personal_projects/rsna_knee_kaggle || exit 1

MODELS="D:/models/rsna_knee_abnormality"
GATE="$MODELS/experiments/resnet50d_lr_s2026.full.log"
LOG="logs/after_resnet50d_lr.log"

echo "waiting for 'ARM DONE resnet50d_lr' in $GATE" >> "$LOG"
until grep -q "ARM DONE resnet50d_lr" "$GATE" 2>/dev/null; do sleep 30; done
echo "gate cleared at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

bash experiments/arms/ourtrunk44.sh >> "$LOG" 2>&1
echo "ourtrunk44 finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
