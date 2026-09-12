#!/usr/bin/env bash
# after_ghosting.sh -- wait for the KMAR-calibrated motion arm, then take the reading that can price it.
# Armed 2026-09-12 17:31Z, minutes after the arm launched.
#
# THIS SCRIPT TAKES A READING, IT DOES NOT LAUNCH AN ARM, and the reason is the same one after_slot_dropout_hi.sh
# gives: the pre-registered band is CROSS-SITE. holdout909 is drawn from the same decode as training, so
# our own ruler is blind BY CONSTRUCTION to the axis a train-time artefact operator is supposed to move.
# A blind arm queued behind this one would keep the card warm and leave this one unreadable.
#
# TWO CHECKPOINTS IN ONE INVOCATION, one split, one foreign decode:
#
#   ghosting_s2026     the arm. --ghosting 0.040, the MEASURED median per-pixel residual of real patient
#                      motion in KMAR-50K (109 paired volumes, 2026-09-12). Not a dialled severity.
#   band_dense_s2026   its MATCHED CONTROL, and the same control the slot_dropout family declares. Re-measured
#                      here rather than quoted, so the arm's toll and the control's come out of ONE pass.
#                      Its value is -0.0361 (native 0.8578, kept 0.8578, foreign 0.8217), reproduced exactly
#                      by the 17:01Z pass -- if THIS pass does not reproduce it to ~0.0001 the pass is broken
#                      and NEITHER number may be read. The calibration gate, stated before the numbers exist.
#
# READ slot_cost FIRST, ON BOTH ROWS. It serves our own pixels through the shifted code path and must land
# inside +/-0.0001. Anything else means the shifted path is broken and decode_cost is not about pixels at
# all -- the one failure that is otherwise invisible, since a broken slab still prints a plausible number.
#
# THE DECIDING NUMBER is decode_cost(ghosting) against decode_cost(band_dense_s2026) FROM THIS PASS, never
# against the eleven-member pool mean: the control is not in that pool, so reading against it prices eleven
# confounds at once. The pool sd (0.0038) is the SCALE the bands are written in, never the comparator.
#
# WHAT THE BANDS MEAN HERE, and why they are not slot_dropout's. Slot dropout attacks the decode toll by
# making a member indifferent to WHICH slots it is given. Ghosting attacks a different thing entirely --
# robustness to an acquisition artefact this project's cache has never shown the model -- so a null here
# does not touch the dose-response, and a gain here is ADDITIVE to it rather than a second reading of it.
# The two are composable terms, which is the question a later arm carrying both would ask.
#
# THE GATE IS AN OR WITH PID LIVENESS: the artefact alone hangs forever on a silent exit-0 death, and the
# pid alone fires the moment a crash frees the card. The log is named for the CHECKPOINT
# (ghosting_s2026.full.log, basename of --out minus .pt, under $MODELS/experiments/ not logs/) and the
# marker inside it for the RUN (ARM DONE ghosting, run_train's name).
source "$(dirname "$0")/lib.sh"

LOG="$MODELS/experiments/ghosting_s2026.full.log"
CKPT="$MODELS/ghosting_s2026.pt"
FOREIGN="D:/data/external/raptor/corpus"
OUT="$MODELS/ghosting_decode.npz"

arm_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*ghosting*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

while true; do
  grep -aq "ARM DONE ghosting" "$LOG" 2>/dev/null && break
  arm_alive || break
  sleep 30
done
echo "=== GHOSTING GATE OPEN at $(date -u +%H:%MZ) ==="

if [ ! -f "$CKPT" ]; then
  echo "=== NO CHECKPOINT - the arm died, no reading to take ==="
  exit 1
fi

UV_NO_SYNC=1 uv run python -m kneemri.evaluation.foreign_readout \
  --checkpoints \
    "$MODELS/ghosting_s2026.pt" \
    "$MODELS/band_dense_s2026.pt" \
  --labels "$LABELS" \
  --foreign "$FOREIGN" \
  --device cuda \
  --per-study "$OUT"

echo "=== GHOSTING DECODE READING DONE at $(date -u +%H:%MZ) - per-study at $OUT ==="
