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
# NO SEED TWINS. rank24_s7 differs from its sibling only by seed and was in this list to "settle
# whether seed diversity is worth anything". It is out: seeds and folds are the same move played
# twice, both are banned here, and an arm that measures a banned method still spends the loop on it.
# The mechanism argument is also already settled without running anything — the noise floor is the
# holdout's SAMPLING error (predicted 0.0053, measured 0.0062, knee-2rx), so re-running the same
# recipe cannot move what the same 881 studies with the same 62 Fractures will report.
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

# A REFUTED ARM IS STILL A MEMBER. slices16_ws peaked at 0.8452 against the control's 0.8457, so 16
# cached slices is refuted as an IMPROVEMENT - and that says nothing about whether it fails on the same
# studies as the others, which is the only thing that matters for an average. It reads a different
# depth of stack, so it is exactly the kind of member worth having. Its inference costs a second decode
# (pixel_group includes the slice count), which is a real cost and the reason to check it earns a place
# rather than assume it.
# res392 reads the 448 px cache and every other member reads 336, which member_curve could not mix
# until it learned to group checkpoints by cache and read each one the way it was fitted. It is also
# our best single member (0.8523), so a curve without it was ranking subsets of the second team.
MEMBERS=(
  "$PROCESSED/res392_ws_s2026.pt"
  "$PROCESSED/sliceattn_s2026.pt"
  "$PROCESSED/wsfold0_e6.pt"
  "$PROCESSED/slices16_ws_s2026.pt"
  "$PROCESSED/lr5x_d6_s2026.pt"
  "$PROCESSED/max12_s2026.pt"
  "$PROCESSED/max20_s2026.pt"
  "$PROCESSED/large_deep_rank24.pt"
)
require "${MEMBERS[@]}"

uv run python -u -m kneemri.member_curve --overlap --labels "$LABELS" --checkpoints "${MEMBERS[@]}" 2>&1 \
  | tee "$LOGS/diversity_curve.log"
echo "ARM DONE diversity_curve"
