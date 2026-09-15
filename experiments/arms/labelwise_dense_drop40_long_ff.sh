#!/usr/bin/env bash
# bead: knee-sexix
# unmeasurable: labelwise_dense_drop40_long_ff fits --fold -1, so it holds no study back and no ledger row
#               can price it (knee-s8kim). Its holdout TWIN labelwise_dense_drop40_long_s2026 reads 0.9048
#               ledger75 (converged, peak e23 of 24), the third drop40 seat of tripleseat_v2.
#
# WHAT THIS ARM IS
#   `labelwise_dense_drop40_long`'s line plus `--fold -1 --ema`, the third seat of the refit road that
#   `labelwise_ext_drop40_long_ff` opened. Run AHEAD of fullseat_v1's board reading (ref 56246750) because
#   the card is otherwise idle (owner 2026-09-15: "gpu is free"); its reading is fullseat_v1's band.
#
# PRE-REGISTRATION (board). fullseat_v1 >= 0.947: this seat swaps for its twin in the next entry.
#   fullseat_v1 <= 0.946: this checkpoint is parked, not shipped, and the refit road stops.
#
# deliberate: --fold -1 --ema — the champion's own fit, matched to labelwise_ext_drop40_long_ff.
# deliberate: --slot-dropout 0.40 --epochs 24 — imported operating point (knee-yotj8), inherited from the twin.
# deliberate: --slices 20 --calibrated-targets --hold-annotations — inherited from the twin.
# deliberate: --vram-gb 22 — owner VRAM cap 50-70% of the card; the twin's 28 predates DeviceBudget.needed().
#
# CARD. ~25% more studies per epoch than the twin, ~3 h (unverified). Nothing else of ours on the card.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

FIT="--fold -1 --ema"

run_train labelwise_dense_drop40_long_ff \
  $RECIPE $FIT --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --vram-gb 22 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --hold-annotations --slot-dropout 0.40 \
  --labels "$LABELS" --out "$MODELS/labelwise_dense_drop40_long_ff.pt"