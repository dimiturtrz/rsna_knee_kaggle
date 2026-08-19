#!/usr/bin/env bash
# Decode a pixel cache: build_cache.sh [size] [slices] [scheme] [workers]. RUN IT ALONE - it must not
# share the machine with a training arm.
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

# THE SCHEME DECIDES THE CACHE'S SHAPE, so it belongs here. It was missing: this arm could only ever
# build canonical six-slot caches, while extended_slots.sh waits on a nine-slot one - a chain that
# would have blocked forever on a cache nothing in the tree could produce. `kneemri.corpus.build_cache` has
# taken --scheme since the extended slots landed; only this wrapper did not pass it.
SCHEME="${3:-canonical}"

# Ten workers held a 63.7 GB memmap beside a trainer and the OS killed both. Six is the standing
# default because this arm's failure mode is not a slow build, it is a lost one - there is no resume.
WORKERS="${4:-6}"

# CROP is the physical extent and CENTRE is the point it is taken about (knee-yrf). Both default to
# the historic reading, which every cache on disk and every member fitted on one already means.
CROP="${5:-130}"; CENTRE="${6:-image}"

# THE BAND IS A READING AXIS AND IT WAS THE NEXT ONE THIS WRAPPER COULD NOT PASS, exactly as --scheme
# was above: `kneemri.corpus.build_cache` has taken --band-low/--band-high since the wide band landed,
# a trainer does not decode on demand, and so band_full - the whole 0.00-1.00 stack - smoked into
# `FileNotFoundError: images_336px_12sl_0-1band.npy` with nothing in the tree able to write that file.
# Defaults are the historic pair, so every existing call keeps its cache name.
BAND_LOW="${7:-0.2}"; BAND_HIGH="${8:-0.8}"

# ASKED FOR, never rebuilt here. This used to reassemble StudyCache's key in bash, so the completion
# check below tested a filename the builder had stopped writing the moment a reading axis was
# added - and the arm's own comment claimed the gate and the artefact could not drift. They could.
KEY="$(uv run python -m kneemri.corpus.build_cache --print-key \
  --image-size "$SIZE" --slices "$SLICES" --scheme "$SCHEME" --crop-mm "$CROP" --centre "$CENTRE" \
  --band-low "$BAND_LOW" --band-high "$BAND_HIGH")"

uv run python -u -m kneemri.corpus.build_cache \
  --image-size "$SIZE" --slices "$SLICES" --scheme "$SCHEME" \
  --crop-mm "$CROP" --centre "$CENTRE" --band-low "$BAND_LOW" --band-high "$BAND_HIGH" \
  --workers "$WORKERS" 2>&1 \
  | tee "$LOGS/cache_${KEY}.log"
status="${PIPESTATUS[0]}"
[ "$status" -eq 0 ] || { echo "ARM FAILED build_cache_${KEY} (exit $status)"; exit "$status"; }
[ -f "$PROCESSED/mask_${KEY}.npy" ] || { echo "ARM FAILED build_cache_${KEY} (no mask written)"; exit 1; }
echo "ARM DONE build_cache_${KEY}"
