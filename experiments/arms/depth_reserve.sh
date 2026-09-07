#!/usr/bin/env bash
# What DEPTH is worth to a whole-stack member at serve time, on a twin that held fold 0 out.
#
# The champion's seats are all 20-slice and all dark (`--fold -1`); `band_dense_s2026` is the readable
# twin of that recipe, so a re-serve at 24 and 40 slices prices the depth axis on the only member that
# can be priced at all. Band and read size are the member's own (0.1-0.9, 336) — depth is the variable,
# and a second one would make the reading unattributable.
set -u
cd "$(dirname "$0")/../.."
MODELS="D:/models/rsna_knee_abnormality"

for depth in 24 40; do
  .venv/Scripts/python.exe -m kneemri.inference.holdout_backfill \
    --checkpoints "$MODELS/band_dense_s2026.pt" --served-slices "$depth" --served-band 0.1 0.9
  echo "QUEUE depth $depth exited $? $(date -u +%FT%TZ)"
done
echo "ARM DONE depth_reserve $(date -u +%FT%TZ)"
