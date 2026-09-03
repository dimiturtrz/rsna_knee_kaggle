#!/usr/bin/env bash
# cnx_mid_rate — the third point on a two-point bracket whose endpoints BOTH fail, in opposite directions.
#
# WHAT IS ACTUALLY OPEN. knee-jk3's note records the finding and then declines to act on it: "cnx_dense
# gained +0.0259 from a fitted rate". That sentence flatters 4e-4. Read the two trajectories instead of
# the two endpoints:
#
#   cnx_dense       4e-5   peaks at e11 of 11 with the training loss still falling — it never converged
#   cnx_own_recipe  4e-4   peaks at e6 of 12 and then declines — it converged and then overfitted
#
# An undertrained point and an overfitted point do not bracket a fitted rate; they bracket an INTERVAL
# nobody has sampled. 1.2e-4 is the geometric mean of the two, which is the only defensible single sample
# of a rate interval (rate space is multiplicative — the arithmetic mean 2.2e-4 sits 4.6x above the low
# end and 1.8x below the high one, and would re-ask the question 4e-4 already answered).
#
# ONE FLAG MOVES. Every other flag is byte-identical to cnx_own_recipe, so the delta is the rate and
# nothing else. cnx_own_recipe is therefore a MATCHED control that already exists and costs no card.
#
# THIS IS AN INSTRUMENT AND ITS EXPECTED SCORE IS ~0. Say it before the number rather than after. Even a
# clean win here lands the trunk near ~0.83 against base_dense 0.8512, and forty displacement tests
# (experiments/README.md:3086) say a member at that quality does not move the peak set. The arm is NOT a
# seat candidate and must not be reported as one.
#
# WHAT IT BUYS INSTEAD, which is why it runs. Every non-DINOv2 trunk this project has ever trained —
# cnx, raddino, vjepa — was given 4e-5, a rate fitted over ~189 DINOv2-family runs and handed on
# unexamined. That is a FAULTY TRAINING suspect standing behind the whole "foreign architectures are
# worse" conclusion, and it has never been cleared. The finding this arm can deliver is about the
# HARNESS, not about ConvNeXt:
#
#   - lands well above 0.8186 → the foreign-trunk numbers on record are rate artefacts, at least partly.
#     The conclusion "our family beats every other trunk" is then resting on a knob, and every foreign
#     arm's number is downgraded to "at a rate that was never fitted for it".
#   - lands at or below 0.8186 → 4e-4 was not merely the high end, it was near the optimum, the interval
#     holds nothing, and the harness suspect is CLEARED. The foreign-trunk conclusion then stands on its
#     own and no future foreign arm owes a rate sweep.
#
# Both outcomes are worth having and only one of them is good news, which is what makes it an instrument.
#
# PRE-REGISTRATION, against cnx_own_recipe = 0.8186 and the 0.0062 holdout floor:
#   >= 0.8310   the interval holds real score. The rate suspect is LIVE; a foreign trunk's rate is fitted
#               before its quality is quoted, and cnx_dense/raddino/vjepa are all re-readable as
#               under-rated rather than under-powered.
#   0.8125-0.8309  flat against its control. The optimum is broad and near 4e-4; the suspect is cleared at
#               the resolution that matters and no foreign arm owes a sweep.
#   <  0.8125   1.2e-4 is worse than both neighbours, which a single smooth optimum cannot produce. Read
#               the trajectory before reading the score: that shape means a schedule interaction with
#               one_cycle, not a rate finding, and the arm reports a harness fault rather than a number.
#
# TRAJECTORY IS PART OF THE READING, not a footnote. The peak EPOCH answers the question the score cannot:
# a peak at e11-e12 with the loss still falling says undertrained whatever the score is, and a peak before
# e6 says this rate overfits faster than 4e-4 did.
#
# bead: knee-ft7m
#
# THE FOUR MOVED FLAGS ARE INHERITED, DECLARED. They are cnx_own_recipe's, not this arm's thesis; the
# checker cannot tell an inherited deliberate flag from a fresh one, so they are re-declared here.
# deliberate: --train-size
# deliberate: --backbone-lr
# deliberate: --batch
# deliberate: --accumulate
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
# control  cnx_own_recipe = 0.8186 (declared 2026-09-03 from this header own prose, knee-sgzgc)
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train cnx_mid_rate \
  $RECIPE --epochs 12 --batch 1 --accumulate 6 --slice-pooling max --whole-stack \
  --variant cnxbase --backbone-lr 1.2e-4 --slices 20 --band-low 0.1 --band-high 0.9 --train-size 448 \
  --vram-gb 14 \
  --labels "$LABELS" --out "$MODELS/cnx_mid_rate_s2026.pt"
