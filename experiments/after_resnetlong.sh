#!/usr/bin/env bash
# after_resnetlong.sh — wait for `resnet_224_short`, read its gate, and run `resnet_224_long` only if it
# passes. Armed 2026-09-09 19:29Z while the short arm was on its first epoch.
#
# THE GATE IS THE LONG ARM'S OWN, NOT A NEW DECISION: `resnet_224_long.sh` says "DOES NOT RUN unless
# `resnet_224_short` reads >= 0.80", because 48 epochs on a trunk whose 12-epoch twin missed the band is
# four times the card spent confirming a recipe defect. This script only mechanises that sentence.
#
# THE LOG IS NAMED FOR THE CHECKPOINT AND THE MARKER IS NAMED FOR THE RUN, and they differ here by the
# seed suffix: the file is `resnet_224_short_s2026.full.log` (basename of --out minus .pt) and the line
# inside it reads `ARM DONE resnet_224_short` (run_train's name). Greping the checkpoint spelling in the
# marker waits forever with an idle card behind it, and `queue_check` reports OK the whole time.
#
# AND THE GATE IS AN OR WITH PID LIVENESS, because the artefact alone hangs forever on a silent exit-0
# death while the pid alone fires the next arm the instant a crash frees the card.
source "$(dirname "$0")/lib.sh"

LOG="$MODELS/experiments/resnet_224_short_s2026.full.log"
CKPT="$MODELS/resnet_224_short_s2026.pt"

short_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*resnet_224_short*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

while true; do
  grep -aq "ARM DONE resnet_224_short" "$LOG" 2>/dev/null && break
  short_alive || break
  sleep 30
done
echo "=== SHORT GATE OPEN at $(date -u +%H:%MZ) ==="

if [ ! -f "$CKPT" ]; then
  echo "=== NO CHECKPOINT - the short arm died, long NOT launched ==="
  exit 1
fi

BEST=$(UV_NO_SYNC=1 uv run python -c "
import json
rows=[json.loads(l) for l in open(r'$MODELS/resnet_224_short_s2026.metrics.jsonl')]
print('%.4f' % max(r['auc'] for r in rows if 'auc' in r))
")
echo "=== SHORT BEST $BEST (gate 0.80) ==="
awk -v b="$BEST" 'BEGIN { exit (b >= 0.80) ? 0 : 1 }' || {
  echo "=== BELOW GATE - long NOT launched, the CNN lane closes on its short arm ==="
  exit 1
}

bash "$(dirname "$0")/arms/resnet_224_long.sh"
