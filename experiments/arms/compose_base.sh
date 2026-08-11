#!/usr/bin/env bash
# The shipping candidate: both surviving levers on the better encoder.
#
# Deliberately moves two things at once. That is normally the mistake this project keeps paying for,
# and it is allowed here for one reason: both levers are ALREADY isolated on small, so nothing is being
# attributed by this arm. What is wanted is the number a submission would carry.
#
# control  base + ranking          0.8283
# adds     encoder rate 4e-5       isolated at +0.0051 (rate_isolate)
# adds     max slice pooling       isolated at +0.0038 (slice_count_maxpool, arm Q)
# If the levers add, this lands near 0.837.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train compose_base \
  --fold 0 --epochs 24 --one-cycle --weight-floor --rank-weight 1.0 \
  --variant base --backbone-lr 4e-5 --slice-pooling max \
  --labels "$LABELS" --out "$PROCESSED/compose_base_s2026.pt"
