#!/usr/bin/env bash
# bead: knee-fnhbs
# reading: labelwise_drop_drop40_long_s2026.holdout.npz
#
# WHAT THIS ARM IS
#   `labelwise_drop`'s byte-identical line with `--slot-dropout 0.40` and `--epochs 12 -> 24` added.
#   Nothing else moves. Its own `--stochastic-depth 0.05` is inherited unchanged.
#
# THE ARGUMENT IS ADDITIVITY, AND IT IS DELIBERATELY NOT THE OPERATOR
#   By the time this arm runs the operator question is answered on three bases, and a fourth would be a
#   repeat if that were the question. It is not. Measured 2026-09-13, full 909, LogitAverage, no
#   selection:
#
#     core six                                        0.9050
#     + labelwise_depth24_drop40_long_s2026           0.9070  (+0.0020)
#     + labelwise_ext_drop40_long_s2026               0.9067  (+0.0017)
#     + BOTH                                          0.9081  (+0.0031)
#
#   +0.0031 against +0.0020 and +0.0017 separately is NEAR-ADDITIVE: the two upgraded members are not
#   redundant with each other, so each further upgraded base is a fresh SEAT rather than a copy of one
#   already held. That is what this arm buys, and knee-0v8yz says member quality at count 1 is the only
#   thing that has ever moved this board 0.005+.
#
# WHY THIS BASE
#   `labelwise_drop` is the fourth and last twelve-epoch labelwise member in the family (0.8984 @e10)
#   and it carries a regulariser none of the other three has. That makes its upgraded member the one
#   least likely to correlate with the other three, which is exactly what the additivity argument needs.
#
# PRE-REGISTRATION — read on the ENSEMBLE, not on the solo, because the solo is not the claim
#   >= +0.0015   marginal over the seven that precede it. Seats it; ship the eight or nine.
#   +0.000..+0.0014
#                directional, no slot.
#   <= 0.0000    additivity is exhausted at three upgraded bases and the family closes. A real closure,
#                and worth the card to have it.
#   solo floor   >= 0.8984 (its own base), or the transfer failed HERE and the ensemble read is void.
#   peak         if this arm peaks ON ITS LAST EPOCH the budget is still binding at 24 and no verdict
#                may be read. Record it and stop.
#
# READ PEAK-TO-PEAK ONLY. $RECIPE carries --one-cycle; the anneal stretches with the budget.
#
# deliberate: --slot-dropout 0.40 — imported operating point (knee-yotj8), not a swept knob.
# deliberate: --epochs 24 — imported from the pair that measured the operator, not a swept knob.
# deliberate: --stochastic-depth 0.05 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets
#             --hold-annotations — inherited verbatim from labelwise_drop. Two flags move, not seven.
#
# PRICE: ~0.8 h. P(clears +0.0015 marginal) ~ 0.45.
#
# control  labelwise_drop_s2026 = 0.8984 ledger75 (on disk, 12-epoch, same recipe line), and the
#          SEVEN-member ensemble measured above. No control run is owed.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train labelwise_drop_drop40_long \
  $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --stochastic-depth 0.05 --slot-dropout 0.40 \
  --labels "$LABELS" --out "$MODELS/labelwise_drop_drop40_long_s2026.pt"
