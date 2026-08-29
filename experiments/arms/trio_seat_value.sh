#!/usr/bin/env bash
# trio_seat_value.sh — price every packaged member of the trio's own pixel group against trioseat_v1.
#
# bead: knee-vfnr
#
# NOT A TRAINING ARM. It spends no card time on a fit; it reads checkpoints that already exist through
# their encoders and asks one question offline. It is shaped as an arm so `after.sh` can queue it behind
# a trainer, because the only thing it needs the card for is inference throughput.
#
# WHY THIS QUESTION, AND WHY IT IS FREE. `trioseat_v1` scored 0.938 with three seats sharing ONE decode
# — matched on slices (20), band (0.1-0.9), image size (336) and backbone (dinov2-small), differing only
# in supervision treatment. That is +0.013 over the single seat for the decode cost of one member, and
# the only board entry that separates treatment from decode. It also refuted a pre-registration of
# 0.925-0.929 that had read `knee-nas5`'s pixel-group collapse result as if it constrained SEATS; it
# constrains pixel GROUPS, and three seats in one group are not one seat.
#
# The lane that opens is: how many treatment seats does that one decode hold? Twenty-eight members
# already packaged in `dimiturnt/kneemri-members` sit at (20 slices, 0.1-0.9, 336px). A wider one-decode
# roster therefore costs ZERO card time and ZERO packaging — a kernel push and a slot. What it does not
# cost, it also does not justify: a slot is spent only if a member prices positive here first.
#
# HOW IT IS READ, written before the number. `seat_value --against trioseat_v1` reports each candidate's
# marginal contribution to that roster on the fold-0 holdout.
#   * Any candidate at or above the 0.0062 holdout floor. Take the best two or three into a wide-trio
#     kernel; the lane is real and the one-decode roster is not yet saturated at three.
#   * Every candidate inside the floor. Three treatments already span what this decode can see, and the
#     efficiency-track entry stays at trioseat. Spend the next slot on decode distance instead.
#   * Every candidate NEGATIVE. Treatment diversity saturates below three and trioseat's 0.938 is a
#     board-noise reading of the pair's 0.936 rather than a seat result — which would reopen, not close,
#     the pre-registration that missed.
#
# THE OFFLINE NUMBER IS NOT THE BOARD NUMBER at any branch. `knee-nas5` measured this axis transferring
# at 2.4x (0.0059 offline arriving as 0.014 on the board), the only ensemble axis in this project that
# amplifies rather than compresses. A candidate inside the floor offline is therefore not obviously dead
# — but it is not evidence either, and a slot is not spent on one.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

# The three seats trioseat already holds are excluded: `--against` supplies them.
CANDIDATES=(
  band_dense_s2026 band_dense_long_s2026 base_dense_s2026 grade_ladder_s2026 band140_s2026
  aug_long_s2026 small_accum_s2026 severity_weight_s2026 small_batch6_s2026 aprior_dense_s2026
  focal_dense_s2026 margin_seat_s2026 sided_pool_s2026 v3base_dense_s2026 calib_dense_hold_s2026
  finding_calib_dense_s2026 mae_matched_s2026 merged_dense_cf
  grade_ladder_ff through_dense_ff base_dense_ff small_batch6_ff final_ws_e12
)

paths=()
for name in "${CANDIDATES[@]}"; do
  paths+=("$PROCESSED/$name.pt")
done

python -m kneemri.kaggle.seat_value --against trioseat_v1 --candidates "${paths[@]}"
