#!/usr/bin/env bash
# labelwise_drop.sh -- the 0.75 ruler's top family crossed with the only regulariser still climbing.
#
# bead: knee-oinpc
# reading: labelwise_drop_s2026.holdout.npz
#
# WHY THIS ARM AND NOT ANOTHER. The roster was re-ranked at cut 0.75 on 2026-09-12 with
# `comp/members.declines` enforced (pool 146, not 329 -- 183 names were already declined in writing).
# The top twelve span 0.0059, so the ruler did NOT break the tie; it REORDERED it. The one family that
# moved is `labelwise_*`: five of the top twelve at 0.75 while ranking 15-40 at 0.5
# (labelwise_depth24 rank 1 at 0.75, rank 18 at 0.5). The champion carries NO labelwise member.
# Meanwhile `dropstack_light` (stochastic depth 0.05) is the only regulariser measured STILL CLIMBING at
# its twelve-epoch wall. The two mechanisms have never been combined.
#
# ONE VARIABLE, DIFFED NOT REMEMBERED. Byte-identical to `labelwise_dense` except `--stochastic-depth
# 0.05` is added. Same recipe, epochs, backbone rate, slices, band, pooling, calibrated targets, held
# annotations, label table and seed. It is also a second one-variable read against `dropstack_light`
# (`--slice-pooling labelwise` vs `max`), so the arm sits on the diagonal of a 2x2 whose other three
# corners are all measured.
#
# PRE-REGISTERED BANDS, against labelwise_dense 0.8613 on the MIDPOINT ruler, because that is the ruler
# the ledger is written in. The 0.75 read (control 0.9034) is reported SEPARATELY and decides nothing.
#   >= 0.8713  the two mechanisms compose. Escalation: labelwise x 24 epochs, and a re-fit of the
#              ensemble around a member that beats the pool's honest top solo.
#   0.8663-0.8712  real but small. The member is seated for ENSEMBLE DIVERSITY -- the champion holds no
#              labelwise seat and the owner's standing directive is that a member which does not
#              outperform solo is still worth testing in an ensemble.
#   0.8614-0.8662  within noise. Stochastic depth is a `max`-pooling effect and does not transfer to
#              labelwise pooling. Seat only if the ensemble fit asks for it; no follow-up arm.
#   <= 0.8613  the regulariser HURTS under labelwise pooling. That closes the cross and says the
#              dropstack gain was pooling-specific, which is what makes this worth running on a null.
#
# PER-LABEL RULE, INHERITED FROM THE MECHANISM. Stochastic depth acts on the trunk, which every label
# reads through, so movement must be BROAD -- dropstack_light moved 10 of 12 labels. A gain concentrated
# on one or two labels is not this mechanism, whatever the mean says. Read the PER-LABEL row first.
#
# HARNESS SUSPECT, ANSWERABLE FROM THE LOG. `StochasticDepth.applied` walks by ATTRIBUTE NAME; an arm
# whose rate reached zero blocks is an ordinary labelwise run wearing this arm's name -- identical
# checkpoint, identical score, identical ledger row. Assert before any verdict: the log must carry
# "stochastic_depth": "0.05" AND `stochastic depth ramped to 0.05 across 12 blocks`.
#
# PRICE: ~56 min (280 s/epoch MEASURED at 20 slices, x 12), no submission slot, and zero extra compute
#        per epoch -- dropping a branch removes activations, never adds them.
# EXPECTED GAIN: P(clears 0.005 over labelwise_dense 0.8613) ~ 0.30. P(clears 1.5%) ~ 0.03 -- this is
#        NOT sold as an improvement. It is sold as an ENSEMBLE SEAT on a family the champion lacks,
#        plus the fourth corner of a 2x2 that closes or opens the pooling-x-regulariser cross.
# VRAM: unchanged from labelwise_dense.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train labelwise_drop \
  $RECIPE --epochs 12 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --stochastic-depth 0.05 \
  --labels "$LABELS" --out "$MODELS/labelwise_drop_s2026.pt"
