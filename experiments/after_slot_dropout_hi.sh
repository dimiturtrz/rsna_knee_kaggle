#!/usr/bin/env bash
# after_slot_dropout_hi.sh -- wait for the dose-2 arm, then take the ONE reading that can price it.
# Armed 2026-09-12 16:19Z, minutes after the arm launched.
#
# THIS SCRIPT TAKES A READING, IT DOES NOT LAUNCH AN ARM -- same shape as after_fieldwide.sh, and for the
# same reason: the pre-registered band is CROSS-SITE, and our own holdout is structurally blind to the
# lever (holdout909 is drawn from the same field distribution as training). Queueing a blind next arm
# behind this one would keep the card warm and leave this one unreadable.
#
# TWO CHECKPOINTS IN ONE INVOCATION, one split, one foreign decode:
#
#   slot_dropout_hi_s2026  the arm. --slot-dropout 0.40.
#   band_dense_s2026       its MATCHED CONTROL, and the SAME control slot_dropout.sh declares in its own
#                          header. Re-measured here rather than quoted, so the dose-2 toll and the
#                          control's toll come out of ONE pass. Its value from 2026-09-12 is -0.0361
#                          (native 0.8578, kept 0.8578, foreign 0.8217) -- if this pass does not
#                          reproduce that to ~0.0001 the pass is broken and NEITHER number may be read.
#                          That is the calibration gate, and it is stated before the numbers exist.
#
# READ slot_cost FIRST, ON BOTH ROWS. It serves our own pixels through the shifted code path and must
# land inside +/-0.0001. Anything else means the shifted path is broken and decode_cost is not about
# pixels at all -- the one failure that is otherwise invisible, since a broken slab still prints a
# plausible number (foreign_readout's own void notice records that happening once, at -0.08).
#
# THE DECIDING NUMBER is decode_cost(slot_dropout_hi) read against decode_cost(band_dense_s2026) FROM
# THIS PASS, never against the eleven-member pool mean -- the control is not in that pool, so reading
# against it prices eleven confounds at once. The pool sd (0.0038) is the SCALE the bands in
# experiments/arms/slot_dropout_hi.sh are written in, never the comparator.
#
# THE GATE IS AN OR WITH PID LIVENESS: the artefact alone hangs forever on a silent exit-0 death, and the
# pid alone fires the moment a crash frees the card. The log is named for the CHECKPOINT
# (slot_dropout_hi_s2026.full.log, basename of --out minus .pt, under $MODELS/experiments/ not logs/)
# and the marker inside it for the RUN (ARM DONE slot_dropout_hi, run_train's name) -- they differ by the
# seed suffix, and grepping the checkpoint spelling in the marker waits forever with an idle card behind
# it while queue_check reports OK the whole time.
source "$(dirname "$0")/lib.sh"

LOG="$MODELS/experiments/slot_dropout_hi_s2026.full.log"
CKPT="$MODELS/slot_dropout_hi_s2026.pt"
FOREIGN="D:/data/external/raptor/corpus"
OUT="$MODELS/slot_dropout_hi_decode.npz"

arm_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*slot_dropout_hi*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

while true; do
  grep -aq "ARM DONE slot_dropout_hi" "$LOG" 2>/dev/null && break
  arm_alive || break
  sleep 30
done
echo "=== SLOT_DROPOUT_HI GATE OPEN at $(date -u +%H:%MZ) ==="

if [ ! -f "$CKPT" ]; then
  echo "=== NO CHECKPOINT - the arm died, no reading to take ==="
  exit 1
fi

UV_NO_SYNC=1 uv run python -m kneemri.evaluation.foreign_readout \
  --checkpoints \
    "$MODELS/slot_dropout_hi_s2026.pt" \
    "$MODELS/band_dense_s2026.pt" \
  --labels "$LABELS" \
  --foreign "$FOREIGN" \
  --device cuda \
  --per-study "$OUT"

echo "=== DECODE READING DONE at $(date -u +%H:%MZ) - per-study at $OUT ==="
