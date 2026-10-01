#!/usr/bin/env bash
# What DEPTH is worth to a whole-stack member at serve time, on a twin that held fold 0 out.
#
# The champion's seats are all 20-slice and all dark (`--fold -1`); `band_dense_s2026` is the readable
# twin of that recipe, so a re-serve at 24 and 40 slices prices the depth axis on the only member that
# can be priced at all. Band and read size are the member's own (0.1-0.9, 336) — depth is the variable,
# and a second one would make the reading unattributable.
set -u
cd "$(dirname "$0")/../.."
MODELS="D:/data/volumetric/mri/models/rsna_knee_abnormality"

for depth in 24 40; do
# THROUGH `uv`: `arm_check` RUNS every arm file to read its flags, and the only thing making that a
# lint is a stub shadowing `uv` alone — a direct interpreter walks past it and the lint becomes a card
# job (measured 2026-09-09). Verified the module imports under `.venv-arm`, which is what
# `experiments/lib.sh` points `uv` at, before the switch.
  UV_NO_SYNC=1 uv run python -m kneemri.inference.holdout_backfill \
    --checkpoints "$MODELS/band_dense_s2026.pt" --served-slices "$depth" --served-band 0.1 0.9
  echo "QUEUE depth $depth exited $? $(date -u +%FT%TZ)"
done
echo "ARM DONE depth_reserve $(date -u +%FT%TZ)"
