#!/usr/bin/env bash
# Waits out latent_mask_ladder.sh, then puts the rate ladder on the card in its place.
#
# The gate is the ARM's OWN interpreter, not a marker line: `latent_mask_ladder.sh` ends either by a rung
# surviving (fine-tune, then exit) or by `require "$SURVIVOR"` failing on a refused last rung, and only the
# second of those writes anything at all. `.venv-arm` is used by no other launcher on this machine, so its
# absence from the process table is the one signal that means "the ladder is off the card" under both.
# `ps -W` is required: MSYS pids are a different number space and `kill -0` reads a live arm as dead.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 1

while ps -W 2>/dev/null | grep -q '\.venv-arm/Scripts/python'; do
  sleep 30
done

echo "mask ladder is off the card at $(date -u +%Y-%m-%dT%H:%M:%SZ); arming the rate ladder"

UV_NO_SYNC=1 uv run python -m kneemri.experiments.card_check || {
  echo "card_check refused — not arming"
  exit 1
}

nohup bash experiments/arms/latent_rate_ladder.sh > logs/latent_rate_ladder.log 2>&1 &
disown
echo "rate ladder armed as pid $!"
