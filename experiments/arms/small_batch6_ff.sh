#!/usr/bin/env bash
# small_batch6_ff.sh — the best member this project owns, retrained on all 4,406 studies.
#
# WHY THIS ARM AND NOT `band_dense_long_ff`. The obvious next full-fit is the one that would complete
# allfit's roster, and its own board entry pre-registers it as UNRUN: the all-data dose read SUB-LINEAR
# from one seat to six (finalfit 0.934 -> allfit 0.937), so the seventh seat is the smallest step on a
# flattening curve, and it costs 24 epochs. This arm converts a DIFFERENT member for a different reason.
#
# WHAT THE BOARD HAS MEASURED FLAT, AND WHAT IT HAS NOT. Three consecutive entries exchanged allfit's one
# fold-0 seat and read 0.937 / 0.937 / 0.936:
#
#   lastseat  band_dense_long_s2026 -> small_batch6_s2026   the BEST member offline (0.8598)   0.937
#   augseat   small_batch6_s2026    -> aug_long_s2026       a member 0.0076 WORSE offline      0.936
#
# So at seven members the seat is not the variable on identity, on quality, or on regularisation. Every
# seat-level lever is closed. The variable that is NOT closed is the one those entries hold fixed:
# there are exactly SIX dark members, so every entry is six-dark-plus-one, and the pool has never had a
# seventh dark member to offer. COUNT at a fixed dose, and DOSE at a fixed count, are the same
# unmeasured axis, and both need one thing: another `_ff` checkpoint. This arm is that checkpoint.
#
# WHY small_batch6 IS THE ONE TO CONVERT. It is the highest-scoring member this project has ever trained
# (0.8598) and it already held the seventh seat in `lastseat` at 0.937, so its fold-0 behaviour in this
# exact ensemble is a MEASURED control rather than an assumption. Converting a member whose seated board
# number is known is what makes the difference readable as dose: same member, same seat, same six
# companions, only `--fold -1 --ema` moves.
#
# THE SINGLE MOVED FLAG. Everything else is byte-identical to `batch_teardown.sh`'s `small_batch6` block
# — epochs, batch 6, slices 20, band 0.1-0.9, pooling, whole-stack, throughput, backbone-lr. `$FIT`
# follows `$RECIPE` so its `--fold -1` overrides the recipe's `--fold 0`, which is how `final_fit_six.sh`
# builds all five of the existing dark members.
#
# PRE-REGISTRATION, written before the number, and it is BOARD-SIDE by construction. A `--fold -1` run
# has seen every holdout study, so this checkpoint gets NO offline score, enters no pair in
# board_transfer, and cannot be greedy-ranked. It is not readable on its own at all; it is readable only
# as the seventh dark seat. Against `lastseat`'s 0.937, which is this same member at this same seat on
# fold 0:
#
#   DOSE STILL PAYS AT SEVEN — >= 0.939. The all-data curve had not flattened after all, the seat that
#     three exchanges read as inert was inert only because it was fold-0, and converting the remaining
#     seats is suddenly the best-priced work on the board. Queue the second conversion immediately.
#   FLAT — 0.936-0.938. The dose curve is saturated at six, which CONFIRMS allfit's sub-linear reading
#     and retires the whole full-fit axis rather than leaving it open. That is worth the 95 minutes on
#     its own: it is the only axis that has ever moved this board, and it would then be closed with a
#     number instead of an extrapolation.
#   WORSE — <= 0.935. Not a seat effect (three exchanges span one tick), so it would indict --ema or
#     all-data training itself at this depth, and the FIRST suspect is the harness, not the axis:
#     confirm the checkpoint mounted and that the kernel named seven members before reading it as a
#     result.
#
# The bands are one board tick wide against a board that has returned 0.936-0.937 four times, which is
# deliberate: this arm is not trying to clear the 1.5% bar alone. It converts the last open structural
# axis into a measured one, and it is the prerequisite for an eight-member all-dark entry that cannot be
# built at all until a seventh dark checkpoint exists.
#
# COST. Twelve epochs at the control's ~361 s, scaled by 4406/3497 studies for the all-data split:
# about 95 minutes. It runs in overnight wall-clock that is otherwise dead, which is the only reason a
# flat-to-one-tick expectation prices as worth running.
#
# bead: knee-rndp
#
# THE ONE MOVED FLAG, DECLARED.
# deliberate: --fold
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

FIT="--fold -1 --ema"

run_train small_batch6_ff \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --batch 6 \
  --labels "$LABELS" --out "$MODELS/small_batch6_ff.pt"
