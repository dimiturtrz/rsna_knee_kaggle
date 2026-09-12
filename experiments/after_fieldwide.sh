#!/usr/bin/env bash
# after_fieldwide.sh -- wait for `fieldwide`, then take the ONE reading that can price it.
# Armed 2026-09-12 14:58Z while the arm was on its first epoch.
#
# THIS SCRIPT TAKES A READING, IT DOES NOT LAUNCH AN ARM. Every other after_*.sh here gates the next
# training run; this one gates a MEASUREMENT, because `fieldwide`'s pre-registered band is cross-site and
# our own holdout is structurally blind to the lever it moves (holdout909 is drawn from the same field
# distribution as training). Queueing a blind next arm behind it would keep the card warm and leave the
# arm unreadable, which is the exact trade `reading_check` exists to refuse. What runs next is decided
# from the number this pass prints, not before it.
#
# FOUR CHECKPOINTS IN ONE INVOCATION, and each one is load-bearing:
#
#   fieldwide_s2026        the arm. --scale-jitter 0.25 --padding-mode reflection.
#   labelwise_drop_s2026   its MATCHED CONTROL, byte-identical recipe but those two flags. The deciding
#                          number is decode_cost(fieldwide) - decode_cost(this), never fieldwide against
#                          the pool mean: that pool is eleven members and the control is not among them,
#                          so reading against it prices eleven confounds at once.
#   slot_dropout_s2026     ALREADY ON DISK, and it tests the THESIS rather than the arm. --slot-dropout
#                          0.25 is the other field every member on disk shares at its identity value, and
#                          it read 0.8592 against `band_dense` 0.8578 -- flat, +0.0014, on the ruler the
#                          thesis says cannot see a shared blind spot. If the thesis is right this member
#                          pays a SHALLOWER toll here despite that flat. If it pays the pool's toll
#                          exactly, the thesis predicts nothing and `fieldwide`'s own number is the only
#                          thing left standing. It costs one marginal member in a pass already running.
#   band_dense_hold_s2026  CALIBRATION. It is in the eleven at decode -0.0360. If this pass does not
#                          reproduce that, the pass is broken and none of the other three numbers may be
#                          read -- which is the failure `foreign_readout`'s own void notice records
#                          happening once already, at -0.08, with plausible numbers logged.
#
# READ slot_cost FIRST, ON EVERY ROW. It serves our own pixels through the shifted code path and must
# land inside +/-0.0001. Anything else means the shifted path is broken and `decode_cost` is not about
# pixels at all -- the one failure that is otherwise invisible, since a broken slab still prints a
# plausible number.
#
# THE GATE IS AN OR WITH PID LIVENESS: the artefact alone hangs forever on a silent exit-0 death, and the
# pid alone fires the moment a crash frees the card. The log is named for the CHECKPOINT
# (`fieldwide_s2026.full.log`, basename of --out minus .pt) and the marker inside it for the RUN
# (`ARM DONE fieldwide`, run_train's name) -- they differ by the seed suffix, and grepping the checkpoint
# spelling in the marker waits forever with an idle card behind it while queue_check reports OK.
source "$(dirname "$0")/lib.sh"

LOG="$MODELS/experiments/fieldwide_s2026.full.log"
CKPT="$MODELS/fieldwide_s2026.pt"
FOREIGN="D:/data/external/raptor/corpus"
OUT="$MODELS/fieldwide_decode.npz"

arm_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*fieldwide*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

while true; do
  grep -aq "ARM DONE fieldwide" "$LOG" 2>/dev/null && break
  arm_alive || break
  sleep 30
done
echo "=== FIELDWIDE GATE OPEN at $(date -u +%H:%MZ) ==="

if [ ! -f "$CKPT" ]; then
  echo "=== NO CHECKPOINT - the arm died, no reading to take ==="
  exit 1
fi

UV_NO_SYNC=1 uv run python -m kneemri.evaluation.foreign_readout \
  --checkpoints \
    "$MODELS/fieldwide_s2026.pt" \
    "$MODELS/labelwise_drop_s2026.pt" \
    "$MODELS/slot_dropout_s2026.pt" \
    "$MODELS/band_dense_hold_s2026.pt" \
  --labels "$LABELS" \
  --foreign "$FOREIGN" \
  --device cuda \
  --per-study "$OUT"

echo "=== DECODE READING DONE at $(date -u +%H:%MZ) - per-study at $OUT ==="
