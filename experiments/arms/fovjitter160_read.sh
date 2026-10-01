#!/usr/bin/env bash
# fovjitter160_read.sh — the DECIDING reading for fovjitter160, taken on the card.
#
# The arm's pre-registration is on decode_cost, not on the native number, so the arm is not read until
# this runs. It goes through `foreign_readout` with the same `--foreign` decode and the same fold the
# eleven-member pool used, so this member's cost is comparable to that pool by construction.
#
# TWO ANCHORS RIDE ALONG, and they are the point of the extra minutes: `crop160_span` (the pool's best
# at -0.0265, the fixed 160 mm field this arm jitters) and `band_dense_hold`. If either anchor does not
# reproduce its recorded cost, the ruler moved and this arm's number says nothing — a reading is only a
# reading against a ruler that still reads the same on a member it has already priced.
#
# THE DEVICE IS THE CARD, never cpu: knee-j99t — on cpu autocast is off, a half-precision member is
# compared against float32, and the residual grows with trunk depth.
#
# bead: knee-10f10
# reading: none -- this IS the reading step; it writes no checkpoint
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

# THROUGH `uv`, FOR TWO REASONS. `arm_check` RUNS every arm file to read its flags, and the only thing
# that makes that a lint is a stub shadowing `uv` alone — an arm naming `.venv/Scripts/python.exe` walks
# past it and the lint becomes a card job (measured 2026-09-09, 32 GB of `foreign_readout` beside a live
# training arm). And this arm is the CONTROL for `carve_adapt_read.sh`, which differs from it in the
# single flag `--adapt`: that arm runs under `uv`, so `.venv` here would make the pair differ in the
# interpreter as well as in the flag. Verified before the change that `kneemri.evaluation.foreign_readout`
# imports under `.venv-arm`, which is what `experiments/lib.sh` points `uv` at.
UV_NO_SYNC=1 uv run python -m kneemri.evaluation.foreign_readout \
  --checkpoints \
    "$MODELS/fovjitter160_s2026.pt" \
    "$MODELS/crop160_span_s2026.pt" \
    "$MODELS/band_dense_hold_s2026.pt" \
  --labels "$LABELS" --foreign D:/data/volumetric/mri/processed/raptor/raptor-corpus --device cuda
