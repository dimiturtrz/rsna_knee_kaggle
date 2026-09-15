#!/usr/bin/env bash
# bead: knee-u15bj
# reading: rampaug_dense_s2026.holdout.npz
#
# WHAT THIS ARM IS
#   `labelwise_dense_drop40_long`'s byte-identical line with ONE change: every regulariser read through
#   the shared ProgressSchedule, climbing from 0 to full strength over the first 30% of the run and
#   descending back to 0 over the last 15%. Geometric widths, slot dropout 0.40 and the rest scale by
#   one multiplier; no per-op schedule.
#
# PRE-REGISTRATION (delta = this arm minus labelwise_dense_drop40_long_s2026 0.9048, ledger75)
#   >= +0.0040   the schedule pays on this seat; candidate recipe default.
#   +0.000..+0.0039  directional. Keep for the pool, spend no slot.
#   <= 0.0000    schedule does not pay at this strength; heavyramp_dense still reads the heavy package.
#   peak         on its LAST epoch = no verdict.
#
# deliberate: --aug-start 0 --aug-mid 1 --aug-end 0 --aug-ramp 0.3 --aug-tail 0.15 — owner's schedule, not swept.
#
# control  labelwise_dense_drop40_long_s2026 = 0.9048 ledger75 (on disk, same recipe line). No control owed.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train rampaug_dense \
  $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --slot-dropout 0.40 \
  --aug-start 0 --aug-mid 1 --aug-end 0 --aug-ramp 0.3 --aug-tail 0.15 \
  --labels "$LABELS" --out "$MODELS/rampaug_dense_s2026.pt"
