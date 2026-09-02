#!/usr/bin/env bash
# merged_span_cf.sh — the empty cell of the champion's decode square, filled with external data.
#
# bead: knee-8sc3
#
# THIS ARM IS NOT A QUALITY ARM, and reading it as one would fail it. The champion's six seats occupy
# three of the four cells of the (12 | 20 slices) x (0.1-0.9 | 0.2-0.8) square:
#
#   (20, 0.1-0.9)  final_ws_cf, grade_ladder_cf, through_dense_cf
#   (12, 0.1-0.9)  calib_span_ff
#   (12, 0.2-0.8)  merged_blend_cf, lr4e-5_ws_cf
#   (20, 0.2-0.8)  EMPTY
#
# The ledger's route statement (experiments/README.md, "Reaching 0.950 therefore needs...") is that the
# member-quality lane ceilings near 0.943 not because terms stop composing but because a roster has a
# fixed number of seats and quality and decode-path diversity spend the same one. What it asks for is a
# seat that is BOTH high-quality and in a pixel group nothing else occupies — external data and a
# different decode path in the SAME member. That is this arm, and the empty cell above is the path.
#
# WHAT IT WILL PROBABLY COST IN MEMBER SCORE, stated first so the number cannot be a surprise.
# `merged_dense_cf` already ran the merged corpus at (20, 0.1-0.9) and read 0.8569, -0.0028 against
# `merged_blend_cf` 0.8597 — the SUBSTITUTION result: a better-labelled corpus and a denser read resolve
# the same fine structure and eat each other. This arm moves only the slice count off `merged_blend_cf`,
# so the same channel collision applies and the expected landing is at or a little below 0.8597. An arm
# whose own pre-registration predicts a flat-to-slightly-negative member is a ~0-expected-score run when
# priced as quality; it is priced here as a SEAT instead.
#
# WHY THE SEAT IS WORTH THE CARD ANYWAY. `knee-nas5` measured pixel-group collapse at 0.0059 offline
# arriving as 0.014 on the board — a 2.4x compression, and the only ensemble axis in this project with a
# board transfer at all. Every other lane's offline gain arrives compressed (calibsix_v1's six seats of
# quality passed through at 23%); this one arrives amplified. A member three thousandths below the seat
# it sits beside, occupying a cell nothing else occupies, is the trade that lane is made of.
#
# ONE VARIABLE against `merged_blend_cf`: 12 slices -> 20, band held at 0.2-0.8. Corpus, label table,
# eval ruler, pooling, rate, epochs, precision and `--calibrated-targets` all held. Against
# `merged_dense_cf` it is the band alone, 0.1-0.9 -> 0.2-0.8, which makes the pair a matched read of
# whether the merged corpus's substitution is with the SLICES or with the SPAN — a question neither
# existing arm can answer alone, and this arm answers it for free on the way to the seat.
#
# HOW IT IS READ, written before the number. Member control `merged_blend_cf` = 0.8597.
#   * >= 0.8597. The substitution is with the SPAN, not the slice count, and the seat is free — take it.
#   * 0.8535 to 0.8596 (within the 0.0062 floor). The expected branch. The member is not the point; the
#     seat is. KEEP if and only if greedy over the roster picks it.
#   * <= 0.8534. The slice count is what the merged corpus substitutes for, `merged_dense_cf`'s -0.0028
#     is explained, and the (20, 0.2-0.8) cell cannot be reached from the merged corpus. The cell would
#     then need a REPORT-labelled member instead, which is a different arm and a worse seat.
#
# THE DECISION RULE IS THE ENSEMBLE, NOT THE MEMBER, at every branch above: greedy over the roster with
# and without it, on the same fold-0 holdout. A member that scores below its neighbour and still enters
# the greedy plateau is the whole hypothesis; a member that scores above it and does NOT enter is a
# refutation of the seat argument, not a confirmation of the arm.
#
# HARNESS CHECK BEFORE READING ANY OF THAT. The printed recipe must name `merged` on "scheme",
# `merged_blend_v1.csv` on "labels", `report_labels_v2.csv` on "eval_labels", `"slices": 20`,
# `band [0.2, 0.8]` and `calibrated_targets true`. The log must load 5,143 studies and hold out 909.
# Three runs have entered this ledger as facts about an architecture when they were facts about a flag
# that never landed.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

MERGED_BLEND="$PROCESSED/labels/merged_blend_v1.csv"
require "$MERGED_BLEND"
require "$LABELS"

run_train merged_span_cf \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.2 --band-high 0.8 \
  --scheme merged --calibrated-targets \
  --labels "$MERGED_BLEND" --eval-labels "$LABELS" --out "$MODELS/merged_span_cf.pt"
