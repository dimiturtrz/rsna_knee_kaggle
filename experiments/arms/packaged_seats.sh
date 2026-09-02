#!/usr/bin/env bash
# packaged_seats.sh — price the free seats that are ALREADY UPLOADED, on the champion's own decodes.
#
# bead: knee-z652
#
# NOT A TRAINING ARM. No fit, no card time beyond inference throughput. It is shaped as an arm so
# `after.sh` can queue it behind a trainer.
#
# WHY IT IS SEPARATE FROM `trio_seat_value.sh`, which prices the same axis. That arm asks which members
# on disk share a read. This one asks which members share a read AND are already in
# `dimiturnt/kneemri-members`. The difference is what a positive COSTS to act on:
#
#   a positive there   -> a dataset push (minutes to hours of upload) plus a kernel edit plus a slot
#   a positive here    -> a kernel edit plus a slot. Nothing else. Today.
#
# That is not a bookkeeping distinction. Five slots a day reset at 00:00 UTC and a push is the step that
# has repeatedly made a slot unusable in the window it was planned for. A ranked list of seats that need
# no push is the list a slot can actually be spent from.
#
# THE COST GROUND, measured by `python -m kneemri.kaggle.roster_reads` over the packaged roster — which
# groups raw checkpoints by the same `Member.decode_key` the kernel serves under, so it cannot disagree
# with what a submission does. The pinned champion `raptorseat_v1` seats six of ours across three of our
# decodes plus the foreign raptor's own:
#
#   read A  canonical / 20 slices / band 0.1-0.9 / 336 — 27 packaged, champion seats THREE
#           (final_ws_cf, grade_ladder_cf, through_dense_cf)
#   read B  canonical / 12 slices / band 0.2-0.8 / 224+336+448 — 19 packaged, champion seats TWO
#           (merged_blend_cf, lr4e-5_ws_cf)
#   span    canonical / 12 slices / band 0.1-0.9 / 336 — 4 packaged, champion seats ONE (calib_span_ff)
#
# So fifty packaged members ride three decodes the champion ALREADY PAYS FOR, and forty-four of them are
# unseated. Every one of those seats costs zero decodes. Read B is the interesting one twice over: it is
# the largest pool AND the only read carrying three resolutions, so its unseated members bring resolution
# diversity as well as treatment diversity for a decode already bought.
#
# HOW IT IS READ, written before the number. `seat_value --against honest12` prices each candidate's
# marginal contribution to that roster on the fold-0 holdout. The bar is the 0.0062 holdout floor
# and NOTHING ELSE — there is no decode budget to trade against here, which is what separates this
# pricing from every seat question this project has asked before.
#   * Any candidate at or above the floor. Take the best into a kernel edit off the champion's own
#     `chosen` tuple and spend one slot. No push, no fit.
#   * Every candidate inside the floor, positive. The reads are saturated at the seats the champion
#     already holds; the free-seat lane is a COST result only and the next slot goes to decode distance
#     (a genuinely new read) rather than to a wider one.
#   * Every candidate NEGATIVE. Seats on a paid read actively harm, which would mean seat value is
#     dominated by member quality rather than by decode diversity — and would sit against `trioseat_v1`
#     (three seats on one decode, 0.938 against a 0.925 single). Read that as a question, not a verdict.
#
# THE OFFLINE NUMBER IS NOT THE BOARD NUMBER. `knee-nas5` measured this axis transferring at 2.4x
# (0.0059 offline arriving as 0.014 on the board), the only ensemble axis here that AMPLIFIES. A
# candidate inside the floor offline is therefore not dead — but it is not evidence, and a slot is not
# spent on one.
# THE CONTROL IS `honest12`, NOT THE PINNED CHAMPION, and the reason is a structural limit rather than a
# preference. Every one of `raptorseat_v1`'s six members is a `_cf` checkpoint — fold -1, fitted on the
# whole corpus including every holdout study — which is exactly why it is the champion and exactly why it
# can never be priced offline. `seat_value` said so itself on 2026-08-29:
#
#   NO SEAT VALUE against raptorseat_v1: 1 of 6 members carry a holdout record, and pricing a marginal
#   contribution needs at least 2
#   unscoreable (fold -1, they have seen every holdout study): calib_span_ff, final_ws_cf,
#   grade_ladder_cf, lr4e-5_ws_cf, through_dense_cf
#
# So `honest12` (0.937 on the board, twelve members, every one fold-0 and scoreable) is the control. It
# is a WEAKER roster than the champion by 0.006 board, which biases every seat value here UPWARD: a seat
# is easier to earn in a weaker roster, and `seat_value`'s own docstring records the shrinking-edge
# effect a rising champion produces. Read a positive as a RANKING over candidates, not as a predicted
# gain against the champion — the ordering is what has transferred here, never the value.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

# READ B, packaged, unseated. `merged_blend_cf` and `lr4e-5_ws_cf` are excluded: the champion holds them.
# `blend_*` are blends of ours rather than fits, listed because a blend is a member to the serve path and
# its correlation with the seated pair is exactly what `seat_value` is for.
READ_B=(
  depth_regions_narrow_s2026 lr4e-5_ws_s2026 res392_ws_s2026 calibrated_targets_s2026
  dinov3_base_lora_s2026 lr4e-5_ws_ff merged_blend_s2026 lr1e-4_ws_s2026 sliceattn_s2026
  intensity392_ws_s2026 lora_r8_v3small_224_s2026 deep24_s2026 blend_ws blend_calib blend_ladder
  blendhold_s2026 base224_s2026
)

# READ A, packaged, unseated. `final_ws_cf`, `grade_ladder_cf` and `through_dense_cf` are the champion's.
# `vjepa_warm_s2026` (0.8337) and `mae_matched_s2026` (0.6735) are in the pool on purpose: a member far
# below the seated ones is the case where alone-quality and seat-value have most often disagreed here
# (calib_span, base_dense), and excluding them by their own control would repeat the error the champion's
# own principles record.
READ_A=(
  calib_dense_hold_s2026 finding_calib_dense_s2026 small_batch6_s2026 band_dense_long_s2026
  small_accum_s2026 band_dense_s2026 grade_ladder_s2026 sided_pool_s2026 through_dense_s2026
  aprior_dense_s2026 merged_dense_cf focal_dense_s2026 severity_weight_s2026 margin_seat_s2026
  base_dense_s2026 aug_long_s2026 v3base_dense_s2026 final_ws_e12 grade_ladder_ff through_dense_ff
  base_dense_ff small_batch6_ff vjepa_warm_s2026 mae_matched_s2026
)

# THE SPAN READ, packaged, unseated. Three candidates only, and the reason to price them separately is
# that this read is the champion's THINNEST: one seat on a whole decode. If a read holds several seats at
# all, the cheapest place to find a second one is the read that currently holds one.
READ_SPAN=(band_span_s2026 calib_span_s2026 calib_span_hold_s2026)

resolve() {
  local out=()
  for name in "$@"; do
    out+=("$MODELS/$name.pt")
  done
  printf '%s\n' "${out[@]}"
}

mapfile -t read_b_paths < <(resolve "${READ_B[@]}")
mapfile -t read_a_paths < <(resolve "${READ_A[@]}")
mapfile -t read_span_paths < <(resolve "${READ_SPAN[@]}")

# THE COST CLAIM, asserted before it is priced. If any of these three lists costs more than one decode,
# a name in it disagrees with the champion on crop, centre, tie, band or spacing and the "free seat"
# framing above is wrong for that list. Cheap to check, and the failure it prevents is a slot spent on an
# entry that serves five decodes while its write-up claims three.
echo "=== cost assertion: each list must be exactly ONE decode ==="
uv run --no-sync --project "$REPO" python -m kneemri.kaggle.roster_reads --max-decodes 1 --candidates "${read_b_paths[@]}"
uv run --no-sync --project "$REPO" python -m kneemri.kaggle.roster_reads --max-decodes 1 --candidates "${read_a_paths[@]}"
uv run --no-sync --project "$REPO" python -m kneemri.kaggle.roster_reads --max-decodes 1 --candidates "${read_span_paths[@]}"

echo "=== read B (12sl, 0.2-0.8, any size) packaged+unseated, against honest12 ==="
uv run --no-sync --project "$REPO" python -m kneemri.kaggle.seat_value --against honest12 --candidates "${read_b_paths[@]}"

echo "=== read A (20sl, 0.1-0.9, 336) packaged+unseated, against honest12 ==="
uv run --no-sync --project "$REPO" python -m kneemri.kaggle.seat_value --against honest12 --candidates "${read_a_paths[@]}"

echo "=== the span read (12sl, 0.1-0.9, 336) packaged+unseated, against honest12 ==="
uv run --no-sync --project "$REPO" python -m kneemri.kaggle.seat_value --against honest12 --candidates "${read_span_paths[@]}"
