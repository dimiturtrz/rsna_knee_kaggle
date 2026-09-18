#!/usr/bin/env bash
# bead: knee-jb0ci
# reading: slicecover36_s2026.holdout.npz
# control: headlr_ship_s2026 = 0.9059 ledger75 (24 slices, same recipe, already on disk)
# deliberate: --batch 6 — 36 slices at 224 px does not fit at 10. batch_teardown measured
# deliberate: batch 6 vs 12 at +0.0020, inside the 0.0062 floor, so this confound is pre-measured null.
set -euo pipefail
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train slicecover36 \
  $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 36 --batch 6 --vram-gb 22 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --hold-annotations --slot-dropout 0.40 \
  --head-lr 2e-4 \
  --labels "$LABELS" --out "$MODELS/slicecover36_s2026.pt"
