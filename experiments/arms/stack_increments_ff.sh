#!/usr/bin/env bash
# stack_increments_ff.sh — the best solo the project owns, moved into the lane the champion ships in.
#
# bead: knee-8ft1
#
# WHY THIS AND NOT A SEATING ARM. Every operation on the champion's SEATING is now measured and inert:
# adding a distant seat read 0.943 (raptorseat), 0.943 (vjepaseat) and 0.942 (fleettwo); reweighting the
# foreign seat bought +0.001 (raptorweight); swapping a seat read 0.944, dead flat against its control
# (holdswap_v1, 2026-08-30). On the offline side three separate candidates priced +0.0002, +0.0005 and
# +0.0000 on the same day. knee-urvf states the consequence: the ensemble tracks best-single + ~0.015,
# that bonus has not grown in fourteen days, and MEMBER QUALITY IS THE ONLY FREE TERM LEFT. So the card
# goes to the best member, not to another arrangement of the ones already seated.
#
# WHY THIS MEMBER. `stack_increments_s2026` is the highest-scoring solo this project has produced. It is
# the only member that distils from `raptor_teacher.npz` and it carries the blend-three corpus, the two
# levers that made the pool's best members, and it is already proven as a recipe — this arm does not
# introduce one.
#
# WHY THE BLIND FIT IS THE WHOLE POINT. Five of the champion's six seats are `--fold -1 --ema`. A fold-0
# member entering that entry pays a fit penalty of roughly 0.008 (the blind-vs-fold-0 board gap), which is
# large enough to eat any quality edge it brought — which is why the pool's best solo has never been
# shippable into the champion. Refitting it blind removes that penalty and makes the quality delta the
# ONLY thing the swap moves. `merged_blend_cf`, the champion's lone fold-0 seat, is the position it would
# take, and that seat pays the penalty today.
#
# ONE MOVED FLAG against `stack_increments`: the fit, `--fold -1 --ema` in place of fold 0. Corpus,
# teacher, distill weight, `--calibrated-targets`, `--hold-annotations`, band 0.1-0.9, slices 20, max
# pooling, backbone rate and epochs are all held, so the resulting member is the known-best recipe and
# nothing else.
#
# `--hold-annotations` is CARRIED, not introduced — it is the control's own flag, and knee-9198 bans
# retraining the ROSTER under it, not carrying it in a refit of one member that already had it.
#
# NO HOLDOUT NUMBER EXISTS FOR THIS RUN, by construction: it has seen every study. Nothing may enter the
# ledger as its score, `seat_value` cannot price it, and `convergence_check` reads its trajectory only.
# The board is its only instrument. That is the cost of the lane, and it is why this arm's job is to make
# the shippable form EXIST rather than to earn a slot by itself.
#
# HARNESS CHECK BEFORE ANY READING. The printed recipe must show `"fold": "-1"`, `"ema": "True"`,
# `blend_three_v1.csv` on "labels", a non-zero distill weight, the raptor teacher path,
# `"calibrated_targets": "True"`, `"hold_annotations": "True"`, `"slice_pooling": "max"`, band [0.1, 0.9]
# and `"slices": "20"`. Three runs have entered this ledger as facts about an architecture when they were
# facts about a flag that never landed.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

BLEND="$PROCESSED/labels/blend_three_v1.csv"
require "$LABELS"
require "$BLEND"
require "$PROCESSED/raptor_teacher.npz"

FIT="--fold -1 --ema"

run_train stack_increments_ff \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --distill-weight 0.3 --teacher-targets "$PROCESSED/raptor_teacher.npz" $FIT \
  --labels "$BLEND" --eval-labels "$LABELS" --out "$MODELS/stack_increments_ff.pt"
