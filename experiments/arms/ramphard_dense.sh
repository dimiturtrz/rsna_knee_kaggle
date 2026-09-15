#!/usr/bin/env bash
# bead: knee-p28we
# reading: ramphard_dense_s2026.holdout.npz
#
# WHAT THIS ARM IS
#   `labelwise_dense_drop40_long`'s byte-identical line with ONE change: example HARDNESS read through the
#   shared ProgressSchedule. Label trust climbs from every weighted cell at the 0.25 confidence floor to
#   the full certainty table over the first 30% of the run, holds, and eases back to half trust over the
#   last 15%. Held annotations stay at weight zero throughout (the superseded --certainty-ramp floored
#   them, which trained early epochs on the gold studies' report labels). No spread file on this seat,
#   so the draw stays uniform and only label trust moves.
#
# PRE-REGISTRATION (delta = this arm minus labelwise_dense_drop40_long_s2026 0.9048, ledger75)
#   >= +0.0040   hardness schedule pays on the dense seat; candidate recipe default.
#   +0.000..+0.0039  directional (certainty_ramp: +0.0008 ledger75 vs band_dense, d89838c8). Keep for pool, no slot.
#   <= 0.0000    easy-to-hard trust does not pay once the dense labels carry calibrated confidence.
#   peak         on its LAST epoch = no verdict.
#
# deliberate: --hard-start 0 --hard-mid 1 --hard-end 0.5 --hard-ramp 0.3 --hard-tail 0.15 — tail to half, not
#   to 0: the last 15% is the one-cycle anneal, and returning to floor weights there would re-weight the
#   uncertain cells while the model settles; half-way keeps confident cells dominant. Owner's shape, not swept.
#
# control  labelwise_dense_drop40_long_s2026 = 0.9048 ledger75 (on disk, same recipe line). No control owed.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train ramphard_dense \
  $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --slot-dropout 0.40 \
  --hard-start 0 --hard-mid 1 --hard-end 0.5 --hard-ramp 0.3 --hard-tail 0.15 \
  --labels "$LABELS" --out "$MODELS/ramphard_dense_s2026.pt"
