#!/usr/bin/env bash
# rampgate — ONE fold-0 member, ~85 min, that decides whether rampsix's 7 h is worth spending.
# bead: knee-fxgm
#
# WHY THIS EXISTS. `rampsix` refits five champion seats with the certainty ramp, and every one of them
# carries `--fold -1` (inherited from `calibsix`, which inherited it from `final_fit_six.sh`). With no
# holdout left, NONE of its five outputs can be scored offline — so the "+0.002 on the first two seats"
# gate written into that arm's header cannot be read at all. It would be seven hours of card with no
# keep/kill available short of a board slot. That is the defect this file fixes.
#
# WHAT IS ALREADY MEASURED, and therefore NOT what this asks. The ramp itself is settled at fold 0:
# `disagreed_ramp_s2026` 0.8574 against its matched control `disagreed_uniform_s2026` 0.8507, +0.0067,
# same signal, same corpus, schedule the only difference. The open question is narrower and it is the
# one that prices `rampsix`: does that +0.0067 TRANSFER to the champion's dense corner, which is where
# three of rampsix's five seats sit and which is a different geometry, a different label policy and a
# different annotation hold from the corner the ramp was measured at.
#
# THE PAIR. `calib_dense_hold` 0.8634 is the champion's best dense seat and is fold 0 — a readable
# control. This member is that member's flags VERBATIM plus the three ramp flags and nothing else, so
# the difference is one mechanism. Same single-flag discipline `calibsix` states and for the same reason:
# a reassembly is a configuration nobody ran.
#
# THE BRANCHES, pre-registered before the card is spent, against the 0.0062 three-seed floor:
#   >= +0.0040  the ramp transfers to the dense corner. rampsix runs its full five seats (7 h) and the
#               arithmetic in its header stands: +0.004-0.008 offline x board transfer 1.55 = +0.006-0.012.
#   +0.0000 to +0.0040  transfers but SMALLER than at the disagreed corner. rampsix runs THREE seats
#               (the dense ones) and not five - 4.2 h, and the two 0.2-0.8 seats stay unspent, since they
#               are the corners calibsix already named as least evidenced.
#   < 0.0000    KILLED for 85 minutes instead of seven hours. The ramp is corner-specific, the +0.0067
#               belongs to the disagreed corner alone, and the uniform-lift lane closes at the champion
#               geometry. Record it; do not re-run it with a different seed (seeds are banned).
#
# COST: one twelve-epoch fit at 336 px, ~425 s/epoch over 4,406 studies = ~85 min. One fifth of the
# commit it gates.
#
# deliberate: --fold 0 (the whole point — rampsix's outputs are unreadable and this one is not)
# deliberate: --sample-spread/--sample-alpha/--sample-ramp (the single mechanism under test)
# deliberate: --calibrated-targets --hold-annotations --slices 20 --band-low 0.1 --band-high 0.9
# deliberate: (all four copied from calib_dense_hold verbatim, so the pair differs in the ramp alone)
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

SPREAD="$PROCESSED/train_spread_honest12.npz"
require "$SPREAD"

RAMP="--sample-spread $SPREAD --sample-alpha 1.0 --sample-ramp 6"
DENSE="--slices 20 --band-low 0.1 --band-high 0.9"

run_train ramp_dense_gate \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $DENSE --calibrated-targets --hold-annotations $RAMP \
  --labels "$LABELS" --out "$MODELS/ramp_dense_gate.pt"
