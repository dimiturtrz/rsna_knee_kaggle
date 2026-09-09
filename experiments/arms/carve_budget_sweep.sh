#!/usr/bin/env bash
# carve_budget_sweep.sh — does ANY adapt budget move the decode cost, or is the toll representation-bound?
#
# WHY IT EXISTS. `carve_adapt_read` (2026-09-09) read a delta of -0.000033 against a same-hour control,
# three orders of magnitude inside its own NULL band. That is not a null on CARVE. Adam displaces a
# parameter by about the learning rate per step whatever the gradient, so the shipped budget --
# `steps = 2` at `lr = 1e-5` -- moves LayerNorm affines of order 1.0 by ~2e-5, a perturbation of 0.002%.
# The reading moved by the amount that arithmetic predicts, so the arm measured its own budget.
#
# WHAT THIS ONE ASKS. The budget is now a flag, so sweep it and let the CARD answer the question M274 put
# to the researcher. The ladder is chosen on |dtheta| ~ steps * lr, which is the quantity that acts:
#
#   lr 1e-3, steps  2  ->  ~2e-3   (100x the shipped budget)
#   lr 1e-2, steps  2  ->  ~2e-2   (1000x)
#   lr 1e-3, steps 10  ->  ~1e-2
#   lr 1e-2, steps 10  ->  ~1e-1   (10% of an affine of order 1.0 -- past this the trunk is being broken,
#                                   not adapted, and a MOVE here reads as damage rather than as CARVE)
#
# ONE MEMBER, NOT THREE, and that is the whole point of running it before a confirmation. The question is
# existential -- does any budget move the number at all -- and one member answers it at a third of the
# card. `fovjitter160_s2026` is the member whose control is the most recently reproduced (-0.0297, twice
# on 2026-09-09, fourth decimal). A point that moves gets re-read on all three; a ladder that does not
# move needs no confirmation, because a mechanism invisible at 1e-1 is not hiding at 1e-2.
#
# THE BRANCHES, before it runs:
#
#   BUDGET CARRIES  — some point moves the decode cost by >= 0.0050 (the arm's own NULL band). Then the
#                     mechanism is live, our shipped budget was simply too small, and D274's quotes decide
#                     whether the setting that moves it is CARVE's or ours.
#   BUDGET DAMAGES  — the cost gets WORSE monotonically with |dtheta|. Then adapting norm affines on one
#                     study is destructive at any size that is felt, which closes CARVE-as-implemented-here
#                     and says the published effect must live in something other than the norm affines
#                     (M274 ask 2: what the paper actually adapts).
#   REPRESENTATION-BOUND — nothing moves by >= 0.0050 anywhere on the ladder, up to a 10% perturbation.
#                     This is the strong version of the branch `carve_adapt_read` pre-registered and could
#                     not fire: the toll survives a perturbation large enough to be felt, so the model-side
#                     lever closes and all weight goes to the data axis (M266).
#
# THE DEVICE IS THE CARD, never cpu: knee-j99t.
#
# deliberate: one member and no seed -- an existential question, not a measurement of a member
# bead: knee-10f10
# reading: carve_budget_*.npz
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

read_budget() {
  local lr="$1" steps="$2" tag="$3"
  echo "=== BUDGET $tag : lr $lr steps $steps ==="
  UV_NO_SYNC=1 uv run python -m kneemri.evaluation.foreign_readout \
    --checkpoints "$MODELS/fovjitter160_s2026.pt" \
    --labels "$LABELS" --foreign D:/kaggle/input/raptor-corpus --device cuda \
    --adapt --adapt-lr "$lr" --adapt-steps "$steps" \
    --per-study "$MODELS/carve_budget_$tag.npz"
}

read_budget 1e-3 2 lr3s2
read_budget 1e-2 2 lr2s2
read_budget 1e-3 10 lr3s10
read_budget 1e-2 10 lr2s10
echo "=== BUDGET SWEEP DONE ==="
