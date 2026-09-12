#!/usr/bin/env bash
# after_maxdose.sh -- when the ghosting reading is on disk and the card is free, run the THIRD dose
# and take its cross-site reading in the same chain.
#
# THE GATE IS THE GHOSTING READOUT'S OWN ARTEFACT, not the ghosting arm's marker. after_ghosting.sh is
# already armed on that marker and follows it with a two-checkpoint foreign_readout that holds the card
# for ~10 minutes; gating this chain on `ARM DONE ghosting` would put a 12-epoch arm on the card while
# that readout is still serving 909 studies through two encoders. The readout's per-study file is the
# only artefact that says BOTH are finished.
#
# THE OR WITH PID LIVENESS is the same rule every watcher here follows: the artefact alone hangs forever
# on a silent exit-0 death, and pid liveness alone would fire the moment a crash frees the card.
source "$(dirname "$0")/lib.sh"

GHOST_NPZ="$MODELS/ghosting_decode.npz"
OUT="$MODELS/maxdose_decode.npz"
FOREIGN="D:/data/external/raptor/corpus"

watcher_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | Where-Object { \$_.CommandLine -like '*after_ghosting*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

while true; do
  [ -f "$GHOST_NPZ" ] && break
  watcher_alive || break
  sleep 30
done
echo "=== MAXDOSE GATE OPEN at $(date -u +%H:%MZ) ==="

if [ ! -f "$GHOST_NPZ" ]; then
  echo "=== NO GHOSTING READING - the chain ahead of this one died. Running anyway: this arm's question"
  echo "=== is independent of ghosting's, and the card would otherwise sit idle."
fi

bash "$(dirname "$0")/arms/slot_dropout_max.sh"

CKPT="$MODELS/slot_dropout_max_s2026.pt"
if [ ! -f "$CKPT" ]; then
  echo "=== NO CHECKPOINT - the 0.55 arm died, no reading to take ==="
  exit 1
fi

UV_NO_SYNC=1 uv run python -m kneemri.evaluation.foreign_readout \
  --checkpoints \
    "$CKPT" \
    "$MODELS/band_dense_s2026.pt" \
  --labels "$LABELS" \
  --foreign "$FOREIGN" \
  --device cuda \
  --per-study "$OUT"

echo "=== MAXDOSE DECODE READING DONE at $(date -u +%H:%MZ) - per-study at $OUT ==="
