#!/usr/bin/env bash
# bead: knee-7gfzv
# reading: heavyramp_dense_s2026.holdout.npz
#
# WHAT THIS ARM IS
#   rampaug_dense's schedule carrying a HEAVIER package: intensity augmentation, ghosting 0.04,
#   stochastic depth 0.05 and mixup 0.2, over 36 epochs. The ramp keeps early fitting fast and the tail
#   lets the anneal settle on clean inputs, which is what should make the heavier plateau affordable.
#   A PACKAGE read by design, not an attribution: a clear lift is ablated afterwards.
#
# PRE-REGISTRATION (delta = this arm minus labelwise_dense_drop40_long_s2026 0.9048, ledger75)
#   >= +0.0040   seat candidate; ablate the package.
#   +0.000..+0.0039  directional. Keep for the pool, spend no slot.
#   <= 0.0000    the heavy package does not pay on this seat even scheduled.
#   peak         on its LAST epoch = no verdict.
#
# deliberate: --aug-start 0 --aug-mid 1 --aug-end 0 --aug-ramp 0.3 --aug-tail 0.15 — owner's schedule, not swept.
# deliberate: --intensity-aug --ghosting 0.04 --stochastic-depth 0.05 --mixup 0.2 --epochs 36 — the owner's heavy package.
#
# control  labelwise_dense_drop40_long_s2026 = 0.9048 ledger75 (on disk, same recipe line). No control owed.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train heavyramp_dense \
  $RECIPE --epochs 36 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --slot-dropout 0.40 \
  --aug-start 0 --aug-mid 1 --aug-end 0 --aug-ramp 0.3 --aug-tail 0.15 \
  --intensity-aug --ghosting 0.04 --stochastic-depth 0.05 --mixup 0.2 \
  --labels "$LABELS" --out "$MODELS/heavyramp_dense_s2026.pt"
