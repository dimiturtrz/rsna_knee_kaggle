#!/usr/bin/env bash
# Decode a pixel cache: build_cache.sh [size] [slices]. RUN IT ALONE - it must not share the
# machine with a training arm.
#
# On 2026-08-11 this build ran beside a trainer and the OS killed BOTH silently: the arm died at epoch
# 7 of 24 and the build at 2200 of 4407, neither leaving an error. 93.6 GB of RAM against a 63.7 GB
# memmap being written and a ~27 GB trainer. VRAM headroom had been checked carefully; system RAM had
# not been checked at all.
#
# There is no resume: open_memmap(mode="w+") recreates the file zeroed, so a kill costs the whole
# decode. That is what makes sharing the machine expensive rather than merely risky.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

SIZE="${1:-448}"; SLICES="${2:-12}"

# Ten workers held a 63.7 GB memmap beside a trainer and the OS killed both. Six is the standing
# default because this arm's failure mode is not a slow build, it is a lost one - there is no resume.
WORKERS="${3:-6}"

uv run python -u -m kneemri.build_cache --image-size "$SIZE" --slices "$SLICES" --workers "$WORKERS" 2>&1 \
  | tee "$LOGS/cache_${SIZE}px_${SLICES}sl.log"
status="${PIPESTATUS[0]}"
[ "$status" -eq 0 ] || { echo "ARM FAILED build_cache_${SIZE}px_${SLICES}sl (exit $status)"; exit "$status"; }
[ -f "$PROCESSED/mask_${SIZE}px_${SLICES}sl.npy" ] || { echo "ARM FAILED build_cache_${SIZE}px_${SLICES}sl (no mask written)"; exit 1; }
echo "ARM DONE build_cache_${SIZE}px_${SLICES}sl"
