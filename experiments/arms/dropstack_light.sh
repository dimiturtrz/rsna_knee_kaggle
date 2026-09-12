#!/usr/bin/env bash
# dropstack_light.sh -- stochastic depth at HALF the rate, the pre-registered follow-up to dropstack_dense.
#
# bead: knee-b9hnw
# reading: dropstack_light_s2026.holdout.npz
#
# THIS IS AN INSTRUMENT, AND IT IS SOLD AS ONE. It is not expected to beat the control. Its job is to
# separate two readings of the same four nulls, which no arm on disk can currently tell apart.
#
# THE QUESTION. Four mechanisms have now been measured at the dense corner at twelve epochs:
#   mixup 0.2          0.8552 @e11   -0.0026, last epoch still climbing
#   EMA                0.8588 @e7    +0.0010, sub-floor, the only positive sign
#   SAM rho=0.05       0.8213 @e6    REFUTED, truncated
#   stochastic depth   (dropstack_dense, this arm's control-mate)
# The control `calib_dense_hold` peaks at e6 (0.8634) and decays to 0.8598 by e11 -- that is genuine
# overfitting, so a regulariser SHOULD have somewhere to work. It does not. The reading I have written
# down is a LAW: at twelve epochs of fine-tuning a pretrained trunk, every regulariser that costs
# convergence loses more than it buys, and EMA is positive precisely because it costs none.
#
# THAT LAW HAS A COMPETING EXPLANATION AND I CANNOT CURRENTLY REFUTE IT: every rate I ran was taken from
# a published schedule calibrated on HUNDREDS of from-scratch epochs. If the rates were simply too large
# for a twelve-epoch fine-tune, then the shape I called a law is an artefact of my own parameter choices
# and the axis is not closed at all. Halving the rate is the cheapest discriminating test that exists.
#
# ONE VARIABLE, DIFFED NOT REMEMBERED. Byte-identical to `dropstack_dense` except `--stochastic-depth`
# 0.1 -> 0.05. Same recipe, epochs, backbone rate, slices, band, calibrated targets, held annotations,
# label table and seed. Both share `calib_dense_hold`'s corner (ledger 0.8634 @holdout909 midpoint),
# which is measured and is not re-run.
#
# PRE-REGISTERED BANDS, and note these are bands on a DISCRIMINATION, not on a gain:
#   >= 0.8634 (>= control)  THE LAW IS WRONG and the rates were the problem. That re-opens the whole
#              regularisation axis, and the escalation is a proper rate sweep (0.025 / 0.075) plus a
#              re-run of mixup and SAM at halved strength. This is the outcome worth the card.
#   0.8557-0.8633  the law HOLDS in shape but the penalty scales with rate, so the mechanism is real and
#              merely mispriced at twelve epochs. No follow-up at this budget; it is re-tested only if
#              the epoch budget ever grows. A member at this level is kept as a diversity seat and
#              nothing is planned around it.
#   < 0.8557 (<= dropstack_dense)  the penalty does NOT scale with rate, which means it is not the rate.
#              THE LAW HOLDS AND THE AXIS CLOSES. No further regularisation arm runs at twelve epochs.
#              This is a real result and it retires a family, which is what an instrument is for.
#
# AND THE SAME PER-LABEL RULE. Stochastic depth acts on the trunk, which every label reads through, so
# any movement must be BROAD. A gain concentrated on one or two labels is not this mechanism, whatever
# the mean says. Read the PER-LABEL row before the verdict.
#
# HARNESS SUSPECT, ANSWERABLE FROM THE LOG. `StochasticDepth.applied` walks by ATTRIBUTE NAME, and an arm
# whose rate reached zero blocks is an ordinary run wearing this arm's name, with an identical checkpoint,
# an identical score and an identical ledger row. The walk RAISES on zero blocks rather than warning, and
# the count is stamped into the recipe. Assert before any verdict: the printed recipe must carry
# "drop_path_rate": 0.05 AND a nonzero "drop_path_blocks" (12 for dinov2-small).
#
# PRICE: ~56 min (measured 286 s/epoch on dropstack_dense at 20 slices x 12 epochs), NO submission slot,
#        and zero extra compute over the control -- no second forward, no parameter, no buffer.
# EXPECTED GAIN: P(beats control) ~ 0.15. Priced as an INSTRUMENT: the payoff is that one of three
#        mutually exclusive readings survives, and two of the three retire work. Under the current law
#        the next regularisation arm would be run on a belief this arm can falsify for 56 minutes.
# VRAM: unchanged from the control. Dropping a branch removes activations, never adds them.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train dropstack_light \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --stochastic-depth 0.05 \
  --labels "$LABELS" --out "$MODELS/dropstack_light_s2026.pt"
