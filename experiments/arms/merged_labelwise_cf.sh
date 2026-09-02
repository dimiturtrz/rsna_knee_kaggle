#!/usr/bin/env bash
# merged_labelwise_cf.sh — external data and a pooling path nothing occupies, in the same member.
#
# bead: knee-vdhk
#
# WHAT THE ROUTE STATEMENT ASKS FOR. `experiments/README.md` (line 4783): the member-quality lane
# ceilings near 0.943 not because terms stop composing but because a roster has a fixed number of seats,
# and quality and decode-path diversity spend the same one — so "reaching 0.950 needs a seat that is
# BOTH high-quality and in a pixel group nothing else occupies, which is external data and a different
# decode path in the same member". `merged_span_cf` answered that on the SLICE/BAND square. This answers
# it on the POOLING axis, which no merged-corpus member has ever varied.
#
# THE CELL. Two greedy seats of the 111-member curve are the two ingredients: seat two is
# `merged_span_cf` (merged corpus, 0.8618, the largest single jump in the curve) and seat three is
# `labelwise_dense_s2026` (`--slice-pooling labelwise`, 0.8613). They are separately seated and have
# never been combined. If the two diversities are independent the combination occupies a cell no member
# does; if they are the same diversity wearing two names, it collapses onto one of them and the curve
# will say so.
#
# ONE VARIABLE against `merged_span_cf`: the pooling, `max` -> `labelwise`. Corpus (`merged_blend_v1.csv`),
# eval ruler (`report_labels_v2.csv`), fold, slices 20, band 0.2-0.8, rate, epochs, precision and
# `--calibrated-targets` all held. So the pair is a matched read of what labelwise pooling does ON the
# merged corpus, which is a question neither parent can answer alone.
#
# WHY FOLD 0 AND NOT A BLIND REFIT, deliberately, when the two arms queued ahead of it are both blind.
# A `--fold -1` member cannot be priced by `RecordMix`, `seat_value` or `decode_entry` — it is only
# readable on the board, and the board costs a slot. This cell has NO fold-0 evidence behind it at all:
# `merged_span_cf` had `merged_blend_cf` and `merged_dense_cf` as neighbours, and this has nothing. Two
# unknowns (does the cell exist, and is the member any good) for the same card time as one. Fold 0 costs
# the identical 100 minutes and returns a number the greedy curve can read, so the cheap proxy gates the
# expensive commit rather than running beside it. If it earns a seat, the blind refit is a later arm.
#
# WHAT IT WILL PROBABLY COST IN MEMBER SCORE, stated first so the number cannot be a surprise.
# `merged_dense_cf` already showed the merged corpus SUBSTITUTES for a denser read (-0.0028 against
# `merged_blend_cf`): a better-labelled corpus and a finer decode resolve the same structure and eat each
# other. Labelwise pooling is a finer decode by the same argument, so the same collision is the expected
# landing. `labelwise_dense` itself was written up as "per-diagnosis pooling fires, and buys nothing at
# the depth built for it". Expect at or slightly below `merged_span_cf`'s 0.8618. This is priced as a
# SEAT, not as quality — same as its parent, and it must be read the same way.
#
# HOW IT IS READ, written before the number. Member control `merged_span_cf` = 0.8618.
#   * >= 0.8618. The two diversities are independent and the cell is free — take the seat outright.
#   * 0.8556 to 0.8617 (within the 0.0062 floor). The expected branch, and NOT a verdict. The member is
#     not the point; the seat is. KEEP if and only if greedy over the roster picks it.
#   * <= 0.8555. Labelwise pooling and the merged corpus substitute for each other, which would be the
#     THIRD instance of the merged corpus colliding with a finer decode (slices, band, now pooling) and
#     would promote that from an observation to a property of the corpus. Then the merged lane's seat is
#     one cell wide and the route statement needs a different external corpus, not a different path.
#
# THE DECISION RULE IS THE ENSEMBLE, NOT THE MEMBER, at every branch: greedy over the roster with and
# without it, on the same fold-0 holdout. A member that scores below its parent and still enters the
# greedy plateau is the whole hypothesis.
#
# HARNESS CHECK BEFORE READING ANY OF THAT. The printed recipe must name `merged` on "scheme",
# `merged_blend_v1.csv` on "labels", `report_labels_v2.csv` on "eval_labels", `"slices": "20"`,
# band [0.2, 0.8], `"slice_pooling": "labelwise"` and `calibrated_targets true`, and the log must load
# 5,143 studies and hold out 909. Three runs have entered this ledger as facts about an architecture
# when they were facts about a flag that never landed.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

MERGED_BLEND="$PROCESSED/labels/merged_blend_v1.csv"
require "$MERGED_BLEND"
require "$LABELS"

run_train merged_labelwise_cf \
  $RECIPE --epochs 12 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.2 --band-high 0.8 \
  --scheme merged --calibrated-targets \
  --labels "$MERGED_BLEND" --eval-labels "$LABELS" --out "$MODELS/merged_labelwise_cf.pt"
