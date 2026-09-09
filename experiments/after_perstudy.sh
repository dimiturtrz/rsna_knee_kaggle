#!/usr/bin/env bash
# Waits out the WHOLE carve chain, then puts the per-study foreign readout on the card in its place.
#
# The gate is the CHAIN'S OWN LAST LINE, not the card, because `after_carve.sh` runs two arms back to
# back and there is a gap of seconds between them where no `.venv-arm` interpreter exists. A watcher
# gated on the card alone would fire into that gap and contend with the second arm — which is exactly
# what turned a three-minute member into eleven once already.
#
# And it is not gated on the chain's PROCESS either, though that reads as the obvious thing: `ps -W`
# prints the executable path and NO argv, so every MSYS script is `/usr/bin/bash` and a grep for a
# script name can never match — the loop would fall through instantly and the guard would be decoration.
# `ps -W` still answers the card, because an interpreter path is a path. Card second, line first.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 1

while ! grep -q 'carve_adapt_read finished' logs/after_carve.log 2>/dev/null; do
  sleep 30
done

while ps -W 2>/dev/null | grep -qE 'venv-arm[\\/]Scripts[\\/]python'; do
  sleep 30
done

echo "the carve chain ended at $(date -u +%Y-%m-%dT%H:%M:%SZ); arming fov_perstudy_read"

UV_NO_SYNC=1 uv run python -m kneemri.experiments.card_check || {
  echo "card_check refused — not arming"
  exit 1
}

bash experiments/arms/fov_perstudy_read.sh > logs/fov_perstudy_read.log 2>&1
echo "fov_perstudy_read finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) with status $?"
