#!/usr/bin/env bash
# after_largefrac.sh — launch the matched-adaptation-fraction arm the moment the confounded one ends.
# The confounded arm is large_labelwise224_dense (--unfreeze-last 6 = a QUARTER of a 24-block trunk);
# large_labelwise224_half is the same arm at 12 blocks, which is the fraction the base control got.
# knee-jzho6.
set -u
cd "$(dirname "$0")/.."
export UV_NO_SYNC=1
stamp() { echo "[$(date -u +%H:%MZ)] $*"; }

# The FILE is named for the checkpoint; the MARKER text is the run_train name and carries no seed.
LOG="D:/models/rsna_knee_abnormality/experiments/large_labelwise224_dense_s2026.full.log"
stamp "waiting for 'ARM DONE large_labelwise224_dense' in $LOG"
[ -f "$LOG" ] || { stamp "ABORT: gate path has no producer"; exit 1; }
until grep -Eq "ARM DONE large_labelwise224_dense|ARM FAILED" "$LOG"; do sleep 60; done
stamp "released: $(grep -Eo 'ARM DONE large_labelwise224_dense|ARM FAILED.*' "$LOG" | tail -1)"

stamp "start large_labelwise224_half"
bash experiments/arms/large_labelwise224_half.sh
stamp "end large_labelwise224_half"
stamp "AFTER LARGEFRAC DONE"
