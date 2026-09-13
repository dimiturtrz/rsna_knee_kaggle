#!/usr/bin/env bash
# bead: knee-evdfd
# reading: band_dense_drop40_long_s2026.holdout.npz
#
# WHAT THIS ARM IS
#   `severity_weight`'s byte-identical line with the objective OFF and two flags imported:
#   `--slot-dropout 0.40` and `--epochs 12 -> 24`. It is the MATCHED CONTROL for
#   severity_drop40_long, and it is written FIRST because the corpus lever was nearly written off
#   against a control that was truncated and operator-free (knee-q1pp0). That subtraction mixed corpus
#   with budget and read +0.0051 for a lever that is worth -0.0012 once both sides converge. The same
#   mistake was available here and this arm is what removes it.
#
# WHY THE OLD CONTROL WILL NOT DO
#   Every one of the twelve `--*-weight` arms is priced against `band_dense_s2026` 0.8578, a 12-epoch
#   operator-free fit. knee-kiss9 measured twelve epochs to bind on two seated recipes (+0.0084 and
#   +0.0088 of holdout from the extra twelve) and knee-q1pp0 measured +0.0111 at ledger75 on the
#   matched plain recipe. A control sitting inside that wall cannot price anything fitted outside it.
#
# WHAT THIS BUYS BEYOND ONE ARM
#   The denominator for the WHOLE family. Eleven other objectives share this recipe line, so any of
#   them that is worth re-taking is re-priced against this arm at no further control cost. That is why
#   the pair is bought together rather than the treatment alone.
#
# PRE-REGISTRATION
#   holdout   expected >= 0.8578 (band_dense's own 12-epoch peak) by the budget argument. A read below
#             refuses the budget story at this corner, which would itself be a finding: it would mean
#             the operator+budget pair pays at 12 slices (base224, +0.0111) and not at 20.
#   peak      if this arm peaks ON ITS LAST EPOCH the budget is still binding at 24 and NO verdict may
#             be read off the pair at all. Record that and stop; do not quote the delta.
#
# deliberate: --slot-dropout 0.40 — imported operating point (knee-yotj8), matched to the treatment.
# deliberate: --epochs 24 — matched to the treatment. Neither is a knob being swept here.
# deliberate: NO --severity-weight — this is the control, and the objective is the one variable.
#
# control  this arm IS the control. band_dense_s2026 and severity_weight_s2026 (both on disk, both
#          12-epoch) give the same subtraction inside the wall, at no extra run.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train band_dense_drop40_long \
  $RECIPE --epochs 24 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --slot-dropout 0.40 \
  --labels "$LABELS" --out "$MODELS/band_dense_drop40_long_s2026.pt"
