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

# THE SECOND PIXEL GROUP, and the reason this arm prices two pools rather than one.
#
# Fourteen fold-0 checkpoints already sit at (12 slices, 0.2-0.8, 224px) — a COMPLETE second one-decode
# group, built months ago as a resolution family and never read as a group. Its decode is the cheapest
# we have: 94 s/epoch against 336's 290, a measured 3.1x, at a member cost of -0.0023 (`native224`
# 0.8502 vs the 336 control `lr4e-5_ws` 0.8525) which is inside the 0.0062 floor. Both self-reported
# rival singles above ours (0.942, 0.938) are at 224.
#
# So this pool is the one lever that moves all three things at once: a NEW pixel group is the decode
# axis `knee-nas5` measured transferring at 2.4x (score), the cheapest decode we own is the efficiency
# entry (prize), and a 3.1x epoch is what every future arm is fitted in (experiment speed). Two of the
# fourteen are packaged; the other twelve cost a dataset push and no card at all.
#
# Priced TWICE on purpose, because the two answers are different questions:
#   * against `trioseat_v1` — can a cheap second group sit beside the treatment group, i.e. do the two
#     one-decode groups compose? This is the efficiency-track roster: two decodes, many seats.
#   * against the pinned champion — is there a seat for a cheap decode in our BEST roster at all? A
#     positive here is a score result and outranks the first; a negative with the first positive means
#     the 224 group is an efficiency play only, which is still a prize we are entered for.
GROUP224=(
  adjacent224_s2026 base224_s2026 blendhold_s2026 confident224_s2026 holdannot_s2026
  labelwise224_s2026 mixup224_long_s2026 mixup224_s2026 mixup224_strong_s2026
  native224_long_s2026 native224_s2026 nocycle224_long_s2026 normcount224_s2026 tissue224_s2026
)

resolve() {
  local out=()
  for name in "$@"; do
    out+=("$PROCESSED/$name.pt")
  done
  printf '%s\n' "${out[@]}"
}

mapfile -t treatment_paths < <(resolve "${CANDIDATES[@]}")
mapfile -t group224_paths < <(resolve "${GROUP224[@]}")

echo "=== treatment group (20sl, 0.1-0.9, 336px) against trioseat_v1 ==="
python -m kneemri.kaggle.seat_value --against trioseat_v1 --candidates "${treatment_paths[@]}"

echo "=== 224 group (12sl, 0.2-0.8, 224px) against trioseat_v1 ==="
python -m kneemri.kaggle.seat_value --against trioseat_v1 --candidates "${group224_paths[@]}"

echo "=== 224 group against the pinned champion ==="
python -m kneemri.kaggle.seat_value --candidates "${group224_paths[@]}"

# THE CHAMPION'S OWN SECOND DECODE, and the reading that made the 224 pool the wrong unit.
#
# `Ensemble.reads` keys a decode on (scheme, slices, tie, reading). SIZE IS NOT IN IT, and neither is
# either pooling — those are model properties, not read properties. Measured, not asserted: a probe
# package of `lr4e-5_ws_cf` (336) plus `native224_s2026` (224) dry-runs as `decode 10s at [224, 336]`,
# ONE read carrying both sizes.
#
# So the unit is not "the 224 family". Thirty-six fold-0 checkpoints share the champion member
# `lr4e-5_ws_cf`'s decode — canonical, 12 slices, band 0.2-0.8 — spread across 224, 336 and 448, and the
# champion seats ONE of them. Seven are packaged. Every other seat on that read is free in decode terms,
# whatever resolution it was fitted at, and the resolution is then diversity we are not paying for.
#
# WHICH OF THEM ACTUALLY SHARE IT, measured rather than assumed. The crop, the centre and the SPACING
# all live inside `reading`, so a member can agree on scheme, slices and band and still cost a second
# read. `python -m kneemri.kaggle.roster_reads --candidates <the 32 below plus four>` groups raw
# checkpoints by the same `Member.decode_key` the kernel serves under, and reported four reads for the
# thirty-two originally listed here: twenty-eight on the champion's, and four splitters —
# `tissue224_s2026` and `centre_tissue_s2026` (centre tissue), `crop80_tissue_s2026` (80 mm) and
# `adjacent224_s2026` (spacing adjacent). Those four are dropped, so this list is now exactly the pool
# whose seats are free; they are still priceable candidates, but each costs a read and belongs to the
# decode-distance question, not this one.
#
# HOW IT IS READ. `seat_value` with no `--against` prices into the pinned champion, which already holds
# this read. Any candidate at or above the 0.0062 holdout floor is a seat that costs NO decode: take it.
# The floor is the whole bar here — there is no decode budget to trade against, which is what separates
# this pricing from every seat question this project has asked before.
GROUP2=(
  confident224_s2026 holdannot_s2026 labelwise224_s2026 mixup224_long_s2026
  mixup224_s2026 mixup224_strong_s2026 native224_long_s2026 native224_s2026 nocycle224_long_s2026
  normcount224_s2026 attend_true_s2026 blendlabels_s2026
  depth_true_s2026 ema_ws_s2026 encoder_layer_9_s2026 focal_pool_s2026 lr2e-4_ws_s2026
  pair_memory_s2026 per_label_loss_s2026 slotmean_s2026 unfreeze4_s2026 res392_lr8e-6_ws_s2026
  base224_s2026 blendhold_s2026 depth_regions_narrow_s2026
  dinov3_base_lora_s2026 lr1e-4_ws_s2026 lr4e-5_ws_s2026
)

mapfile -t group2_paths < <(resolve "${GROUP2[@]}")

echo "=== the champion's own second decode (canonical, 12sl, 0.2-0.8, any size) against the champion ==="
python -m kneemri.kaggle.seat_value --candidates "${group2_paths[@]}"
