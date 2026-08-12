#!/usr/bin/env bash
# Does ensembling by MECHANISM buy what ensembling by fold bought? Measured offline, on members we
# have already paid for, before any of it costs a submission.
#
# THE QUESTION, and why it is now the important one. Five weaker members scored 0.888 on the public
# leaderboard; two members that are +0.021 better EACH scored 0.882. Member quality alone did not
# carry it (knee-lv2). If member COUNT or the DIVERSITY that comes with it is doing the work, we need
# a source of diversity that is not folds and not seeds — and the arms have been producing one all
# along, because every arm is fitted on the SAME fold-0 split and differs by RECIPE.
#
# THE MEMBERS BELOW ARE NOT INTERCHANGEABLE, which is the point:
#   wsfold0_e6         0.8457  whole-stack, per-slice max
#   lr5x_d6_s2026      0.8323  channel stacking, windowed, encoder rate 4e-5
#   max12_s2026        0.8310  per-slice max, windowed
#   max20_s2026        0.8309  per-slice max, windowed, 20 cached slices
#   large_deep_rank24  0.8304  a larger backbone
# A channel-stacked member and a per-slice one are different functions on different pixels, so they
# fail differently — which is the only reason averaging them can beat either.
#
# IT ALSO SETTLES A STANDING CLAIM WITH DATA WE ALREADY OWN. rank24_s7 and rankbase_deep24_s7 differ
# from their siblings only by SEED. `member_curve` scores every subset of each size, so the subsets
# that add a seed twin can be read against the subsets that add a different recipe at the same count.
# That is a direct measurement of whether seed diversity is worth anything next to mechanism
# diversity, rather than an assertion either way.
#
# THE MEMBERS THAT MUST NEVER APPEAR HERE: member_fold1, member_fold3, member_fold4 and any other
# member fitted on a different fold. Their training data is this holdout. That exact mistake once
# reported an ensemble of five at 0.8807 while its own members logged 0.83 — the number was void, and
# nothing about it looked wrong. Old checkpoints do not record their fold, so absence of a fold field
# is NOT evidence of fold 0.
#
# NO GPU TRAINING, only inference over the holdout, so it is minutes rather than hours.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

MEMBERS=(
  "$PROCESSED/wsfold0_e6.pt"
  "$PROCESSED/lr5x_d6_s2026.pt"
  "$PROCESSED/max12_s2026.pt"
  "$PROCESSED/max20_s2026.pt"
  "$PROCESSED/large_deep_rank24.pt"
  "$PROCESSED/rank24_s7.pt"
)
require "${MEMBERS[@]}"

uv run python -u -m kneemri.member_curve --overlap --labels "$LABELS" --checkpoints "${MEMBERS[@]}" 2>&1 \
  | tee "$LOGS/diversity_curve.log"
echo "ARM DONE diversity_curve"
