#!/usr/bin/env bash
# chainfour_ff.sh — the strongest recipe this project owns, refit on the whole corpus so it can take a
# seat beside the champion's six without the dose confound.
#
# bead: knee-jbum4
# unmeasurable: --fold -1 holds nothing back, so this member has no holdout score and no ledger row can
#   price it. That is deliberate and it is the POINT: every one of the champion's six seats is full-fit,
#   and the only reason `proxyseat` has to carry a stated confound is that the three members it adds are
#   fold-held. This arm removes that confound for the strongest of them. Its ruler is the board, through
#   the entry that seats it, and nowhere else.
#
# WHY THIS ARM AND NOT ANOTHER KNOB. Every architecture and serving axis on this project is closed by
# measurement. `chainfour` (fold 0) is the first member to pass `calib_dense_hold`'s 0.8634 in three
# weeks, and it did it on a recipe that did not exist yesterday. Full-fit dose is worth about +0.007
# BOARD on this record (`honest12` 0.937 -> champion 0.944) — the largest measured constant that is not
# already spent. Applying the largest unspent constant to the best recipe is the highest ratio available.
#
# WHAT IS AND IS NOT KNOWN BEFORE IT RUNS. Known: the recipe, because `chainfour` ran it at fold 0 and
# every term's landing was verified in the recipe row. Not known: what a full-fit twin of it contributes
# to an entry, which no offline ruler here can answer and which is exactly why the arm is worth card.
# This is NOT a fold variant run to re-read a number — the fold-0 number is already in hand and this arm
# cannot produce a comparable one. It is the production step that makes the member seatable.
#
# EVERY FLAG IS chainfour's, TWO CHANGED, AND THE CHANGES ARE STATED:
#   --fold -1        replaces the implicit fold 0. This is the whole arm.
#   --out            chainfour_ff_s2026.pt, a distinct file. Never overwrites the fold-0 member, which
#                    is the one every offline reading in `logs/chainfour_ledger.log` is about.
#
# deliberate: --fold -1 (the arm)
# deliberate: --calibrated-targets --labels blend_three_v1 --ema --slice-pooling labelwise (chainfour's
#   four terms, unchanged; changing one would make this a different recipe with no fold-0 reading behind it)
# deliberate: --hold-annotations (kept for flag-for-flag identity with chainfour even though nothing is held)
# deliberate: --eval-labels report_labels_v2.csv is INERT under --fold -1 (no holdout to pin). Kept for
#   the same identity reason blendsix.sh keeps it, not because it does anything here.
# deliberate: --slices 20 --band-low 0.1 --band-high 0.9 --image-size 336 --backbone-lr 4e-5 (the corner, held)
# deliberate: --epochs 12. Under --fold -1 there is no holdout to rank epochs by, so the epoch INDEX
#   stands in (kneemri/training/fitting.py:10) and the budget must match the one whose peak is known.
#   chainfour peaked at epoch 7 of 12; a shorter budget would land the index somewhere else.
#
# COST. ~75 min of card at fold 0's rate, plus whatever the extra ~10% of corpus adds. Zero slots.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
BLEND="$PROCESSED/labels/blend_three_v1.csv"
require "$BLEND"
require "$LABELS"

run_train chainfour_ff \
  $RECIPE --epochs 12 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --image-size 336 --hold-annotations --ema --calibrated-targets --fold -1 \
  --labels "$BLEND" --eval-labels "$LABELS" --out "$MODELS/chainfour_ff_s2026.pt"
