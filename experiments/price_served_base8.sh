#!/usr/bin/env bash
# price_served_base8.sh — the SERVED-SHAPE seat ruler, written down so it stops being dark.
#
# WHY THIS FILE EXISTS. The +0.0178-at-6x that entry 56060097 and its three branch kernels are built on
# was produced by a hand-typed `seat_weight` invocation on 2026-09-06. Only its OUTPUT was kept
# (`logs/seat_weight_served.log`), and that log records neither the members nor the argv — so the number
# that decides the foreign seat's weight could not be reproduced from disk. The member list survived only
# inside a letter. Re-derived 2026-09-06T17:57Z and it reproduces exactly (rank base 0.8835, +0.0178 at
# 6x), which is why this script now IS the derivation.
#
# WHAT THE BASE IS, AND WHY THOSE EIGHT. Seven members of `greedyseat_v1` (board 0.936, ref 55877491 —
# the largest entry all of whose members carry holdout records, so the only base of this size that can be
# priced at all), plus `raptor_v4_swa`, which is the checkpoint `raptor_run.DEFAULT_WEIGHTS` actually
# serves. That eighth seat is what makes this the SERVED shape rather than the greedy one: the shipped
# entry votes a foreign block beside raptor, and raptor supplies part of the cross-family diversity the
# foreign seat is bought for. Leaving it out reads the seat about a fifth too high at every weight.
#
# WHAT IT CANNOT SAY. `raptor_v4_swa` is 98.6% in-sample on these 909 studies by `foreign_overlap`, so its
# own contribution to the 0.8835 base is unpriced and this base is optimistic by an unknown amount. The
# candidate column is not: `foreign_dinov3_matched` is fold-matched (each study scored by the ONE fold
# that never trained on it), which makes every delta below a FLOOR on the served form, since the entry
# serves all five folds per study.
source "${ARM_LIB:-$(dirname "$0")/lib.sh}"

GREEDYSEAT_SEVEN=(
  calib_dense_hold_s2026
  merged_span_cf
  labelwise_dense_s2026
  blendhold_s2026
  finding_calib_dense_s2026
  margin_seat_s2026
  warm_calib_dense_s2026
)

base=()
for name in "${GREEDYSEAT_SEVEN[@]}" raptor_v4_swa; do
  base+=("$MODELS/$name.holdout.npz")
done

candidate="${1:-$MODELS/foreign_dinov3_matched.holdout.npz}"

"$REPO/.venv/Scripts/python.exe" -m kneemri.evaluation.seat_weight \
  --foreign \
  --base "${base[@]}" \
  --candidate "$candidate" \
  --weights 1 2 3 4 5 6 7 8 9 10 \
  --split-half
