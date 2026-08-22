#!/usr/bin/env bash
# Decode a pixel cache: build_cache.sh [any core.data.corpus.build_cache flag] [--workers N] [--dry-run].
# RUN IT ALONE - it must not share the machine with a training arm.
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

# THE DEFAULTS ARE THE HISTORIC READING, which every cache on disk and every member fitted on one
# already means. They are written as the builder's OWN flags and any argument given here is appended
# AFTER them, so argparse's last-wins settles every override and this wrapper holds no parsing.
#
# THAT IS THE POINT, and it is worth more than the shorter call site. Twice this wrapper was the thing
# that could not pass a reading axis the builder had long accepted: --scheme, so extended_slots.sh
# waited on a nine-slot cache nothing in the tree could write, and then --band-low/--band-high, so
# band_full smoked into `FileNotFoundError: images_336px_12sl_0-1band.npy`. Both were an eight-long
# positional list re-encoding a CLI it does not own, which goes stale the day that CLI grows. A
# passthrough cannot: the next axis works here the day the builder learns it, with no edit at all.
DEFAULTS=(--image-size 448 --slices 12 --scheme canonical
          --crop-mm 130 --centre image --band-low 0.2 --band-high 0.8)

# Ten workers held a 63.7 GB memmap beside a trainer and the OS killed both. Six is the standing
# default because this arm's failure mode is not a slow build, it is a lost one - there is no resume.
# It is split out rather than passed through because --print-key does not take it.
WORKERS=6
DRY_RUN=""
READING=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) READING+=("$1"); shift ;;
  esac
done
READING=("${DEFAULTS[@]}" "${READING[@]}")

# ASKED FOR, never rebuilt here. This used to reassemble StudyCache's key in bash, so the completion
# check below tested a filename the builder had stopped writing the moment a reading axis was
# added - and the arm's own comment claimed the gate and the artefact could not drift. They could.
KEY="$(uv run python -m core.data.corpus.build_cache --print-key "${READING[@]}")"

# THE CHEAPEST CHECK THIS ARM HAS: it asks what cache a call would write without decoding one. Every
# failure above announced itself as a wrong or missing filename after the machine had been committed.
[ -z "$DRY_RUN" ] || { echo "would build ${KEY} with ${WORKERS} workers"; exit 0; }

uv run python -u -m core.data.corpus.build_cache "${READING[@]}" --workers "$WORKERS" 2>&1 \
  | tee "$LOGS/cache_${KEY}.log"
status="${PIPESTATUS[0]}"
[ "$status" -eq 0 ] || { echo "ARM FAILED build_cache_${KEY} (exit $status)"; exit "$status"; }
[ -f "$PROCESSED/mask_${KEY}.npy" ] || { echo "ARM FAILED build_cache_${KEY} (no mask written)"; exit 1; }
echo "ARM DONE build_cache_${KEY}"
