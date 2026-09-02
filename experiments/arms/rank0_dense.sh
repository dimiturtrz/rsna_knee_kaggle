#!/usr/bin/env bash
# rank0_dense.sh — turn the pairwise ranking term OFF at the settled operating point, and find out what
# a flag every arm has inherited for thirty-odd runs is actually worth here.
#
# WHY THIS IS A DENOMINATOR AND NOT A KNOB. knee-kdf's whole argument is that this ledger measures every
# proposal against the FULL recipe and has never measured what the recipe's own components carry, so a
# new idea worth +0.01 cannot be judged without knowing whether the piece it competes with is worth
# +0.002 or +0.05. Two of that bead's three components have since been priced by other work — slot
# attention against mean pooling by the unledgered `slotmean` (0.8406, −0.0051, found 2026-08-19) and
# the whole-stack read at +0.0147 — which leaves this one, and leaves it as the last unpriced component
# of the recipe every arm inherits.
#
# IT IS NOT UNMEASURED, AND THAT IS THE POINT. `recipe.sh` justifies the flag as "the only loss change
# positive in 3 of 3 matched arms", and those arms are real: the early era read everything against
# `rank24_s2026` = 0.8272 at 24 epochs, small backbone, before whole-stack and before the band. What has
# happened since is +0.0306 of recipe on top of that control, and this ledger's most repeated single
# finding is that a lever's size does not survive the operating point it was measured at — the slice
# attention that bought +0.0016 at 12 slices bought −0.0009 at 20 (`attend_dense`), the density trade at
# held span reversed direction, and `crop80_span` gave back exactly what the band widening had paid for.
# A term inherited on evidence from a different operating point is an assumption wearing a measurement's
# clothes.
#
# THE MECHANISM SAYS SO MORE SHARPLY THAN THE ANALOGY DOES. The ranking loss is WITHIN-BATCH pairwise:
# its gradient comes from positive-negative pairs formed inside a single batch, so its strength scales
# with the pair count that batch affords, not with the dataset. The ledger already relies on this
# elsewhere — the 448 px rung was rejected in part because dropping to batch 9 "changes the ranking
# term's within-batch pair count and stops it being one variable". This arm runs at batch 12 with 20
# slices and the whole stack; `rank24_s2026` predates the metrics recorder and kept no recipe line, so
# its batch is NOT recoverable and the comparison across eras cannot be made honestly at all. That is a
# second reason to measure it here rather than a measured difference, and it is stated as such.
#
# THE DEGENERATE-ZERO CHECK IS ALREADY DONE AND IT PASSES — knee-kdf's transferable half, stolen from a
# campaign where an ablation knob at zero fell between two modes and produced an empty submission that
# read as "the component is everything". Verified by reading rather than assumed: `train.py`'s
# `if self.rank_weight:` falls through to `Objective(normalise_by_count=..., per_label=...)` carrying the
# SAME flags the ranked branch passes, so `--rank-weight 0` is pure BCE and nothing else moves with it.
# Zero is this mechanism's limit, not a third mode nobody meant.
#
# control  band_dense = 0.8578 (best epoch 8 of 12, 4e-5, 20 slices, band 0.10-0.90, max pooling, whole
#          stack, slot_pooling attend, batch 12, rank_weight 1.0 — read out of the recorded recipe).
#          Single variable: --rank-weight 0. Same schedule, same operating point, same cache.
# seat     comp/champion.json greedy4_v1 = 0.8679 over the 909-study fold-0 holdout.
# cost     about 1.2 h, and slightly less: dropping the term removes the pairwise construction per step.
#
# PRE-REGISTERED, three outcomes:
#   PREDICTION — the peak mean falls below 0.8578 by more than the 0.0062 holdout floor. Then the term
#     is load-bearing HERE and not only in the small era, the last component of the shared recipe has a
#     number on it, and every future loss proposal has a bar to clear that is this size rather than an
#     unknown one. That is the outcome that costs 1.2 h and buys a denominator.
#   REFUTATION — the peak mean is at or above 0.8578. Then the pairwise term buys nothing at 20 slices,
#     whole stack and batch 12; thirty-odd arms have paid its per-step construction on authority
#     inherited from an operating point that no longer exists, and `recipe.sh` should drop it. A flag
#     removed on measurement is a genuine narrowing and a small permanent speedup, and it would also say
#     the +0.0306 of recipe since `rank24_s2026` has ABSORBED whatever the term used to supply.
#   THIRD OUTCOME — the mean sits inside the floor but the seat value against greedy4_v1 is positive.
#     Then the term is not a mean lever but an ERROR-SHAPE lever, which is the reading `deep24` already
#     hints at from the wrong direction: it is the WORST member on disk solo (0.8213, ranking off) and
#     greedy forward selection takes it anyway, because a different objective shapes different errors.
#     `deep24` confounds ranking-off with unfreeze 12, so it cannot carry that claim; this arm is the
#     clean single-variable version of it, and in this outcome the member is priced by seat value rather
#     than by its solo score and the pool gains its second non-recipe diversity axis.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train rank0_dense \
  $RECIPE --rank-weight 0 --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --labels "$LABELS" --out "$MODELS/rank0_dense_s2026.pt"
