#!/usr/bin/env bash
# dropstack_long.sh -- stochastic depth 0.05 at TWENTY-FOUR epochs. The budget question, not the rate one.
#
# bead: knee-ffku5
# reading: dropstack_long_s2026.holdout.npz
#
# WHAT dropstack_light ACTUALLY FOUND, AND IT IS NOT THE MEAN. At 0.05 the arm read 0.8654 against a
# 0.8634 control -- +0.0020, which is noise-sized and is recorded as directional only. The result that is
# not noise is the SHAPE of the two curves:
#
#   calib_dense_hold (control)  peaks e6 0.8634, decays to 0.8598 by e11   -- overfits, budget UNUSED
#   dropstack_dense  (0.1)      0.8595 @e11, e10 == e11                    -- plateaued, converged
#   dropstack_light  (0.05)     0.8654 @e12, STILL CLIMBING at the wall    -- budget BINDING
#
# The control cannot spend twelve epochs; it has nothing to do after six but memorise. The regularised run
# had not turned when the budget ran out. So the product of this mechanism may be EPOCHS rather than the
# two thousandths on the mean -- and epochs are a lever with room, where a rate sweep over a range whose
# whole measured span is 0.006 is not. CLAUDE.md: pick by whether the mechanism could plausibly deliver
# the bar, not by what is nearest to hand. That is why this runs before the 0.025/0.075 sweep the
# dropstack_light header pre-registered.
#
# ONE VARIABLE, DIFFED NOT REMEMBERED. Byte-identical to `dropstack_light` except `--epochs 12` -> `24`.
# Same recipe, backbone rate, slices, band, pooling, calibrated targets, held annotations, stochastic
# depth rate, label table and seed. dropstack_light is measured (0.8654 @holdout909 midpoint) and is not
# re-run.
#
# PRE-REGISTERED BANDS, against dropstack_light 0.8654:
#   >= 0.8754  the regulariser bought a budget the control could never spend, and the gain is real rather
#              than directional. Escalation: 36 epochs, the same rate at the span corner, and a re-fit of
#              the ensemble around a member this much better than the pool's top. This is the outcome
#              worth two hours of card.
#   0.8684-0.8753  real but small. The member is the project's best honest solo and is seated for that,
#              but the budget is not the lever it looked like and no 36-epoch arm follows.
#   0.8655-0.8683  within noise of the twelve-epoch point. The budget was NOT the binding constraint --
#              the RATE was, and the honest next move is the 0.075 point, once.
#   <= 0.8654  a longer budget hurts even a regularised run. That closes BUDGET as a lever at this corner,
#              which retires a family and is what makes this arm worth running even on a null.
#
# AND THE PER-LABEL RULE, WHICH THIS MECHANISM HAS ALREADY PASSED ONCE. Stochastic depth acts on the
# trunk, which every label reads through, so movement must be BROAD -- dropstack_light moved 10 of 12
# labels against dropstack_dense with no single-label concentration. A gain here that sits on one or two
# labels is not this mechanism, whatever the mean says. Read the PER-LABEL row before the verdict.
#
# HARNESS SUSPECT, ANSWERABLE FROM THE LOG. `StochasticDepth.applied` walks by ATTRIBUTE NAME, and an arm
# whose rate reached zero blocks is an ordinary run wearing this arm's name -- identical checkpoint,
# identical score, identical ledger row. The walk RAISES on zero blocks and the count is stamped into the
# recipe. Assert before any verdict: the log must carry "stochastic_depth": "0.05" AND
# `stochastic depth ramped to 0.05 across 12 blocks`, as dropstack_light's did.
#
# AND THE SECOND HARNESS SUSPECT, WHICH IS NEW AT THIS BUDGET. `$RECIPE` carries `--one-cycle`, so the
# schedule STRETCHES to the epoch count rather than repeating: at 24 epochs the anneal is twice as long
# and the run spends far more steps at a high rate. A null here is therefore ambiguous between "the budget
# does not help" and "the stretched schedule is wrong for this corner", and the band above says
# `<= 0.8654` closes the budget lever. Before writing that verdict, check the trajectory for a LATE peak
# that the longer anneal merely delayed -- if the curve is still climbing at e23, the budget is not closed,
# it is still binding and the finding is the same one this arm was built on.
#
# PRICE: ~112 min (279 s/epoch MEASURED at 20 slices, x 24), no submission slot, and zero extra compute
#        per epoch over the control -- no second forward, no parameter, no buffer.
# EXPECTED GAIN: P(clears 0.005 over 0.8654) ~ 0.45, the prior strong because the curve had not turned.
#        P(clears 1.5% over the 0.8634 control) ~ 0.12.
# VRAM: unchanged from the control. Dropping a branch removes activations, never adds them.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train dropstack_long \
  $RECIPE --epochs 24 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --stochastic-depth 0.05 \
  --labels "$LABELS" --out "$MODELS/dropstack_long_s2026.pt"
