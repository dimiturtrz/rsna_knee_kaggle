#!/usr/bin/env bash
# labelwise_dense_ff.sh — greedy seat three, refit blind, so the second swap has two candidates.
#
# bead: knee-zqn8
#
# WHY A SECOND ONE. `merged_span_ff` is the arm that runs first and the one the second seat is expected
# to use. This is the alternate, and it exists because the swap ORDERING does not transfer — only the
# sign does (`knee-7d1`, where the ordering broke on its third point). The step-2 table put
# `labelwise_dense_s2026` at +0.0021, the TOP row, and `merged_span_cf` at +0.0017; both are inside the
# 0.0062 unseparable band, so the table cannot say which of the two is better and neither can I. Having
# both blind means the choice is made on a constraint rather than on a row of an instrument that has
# already been shown not to order.
#
# WHAT SEPARATES THEM, since the ordering does not. `merged_span_cf` sits at greedy seat two with the
# largest jump in the 111-member curve and occupies a decode cell (20 slices, band 0.2-0.8) the champion
# does not; `labelwise_dense_s2026` sits at greedy seat three, is the third-best solo the project owns
# at 0.8613, and its `--slice-pooling labelwise` is a POOLING axis no champion seat carries. Those are
# different kinds of diversity, and which one a six-seat roster wants is an ensemble question, not a
# member question — greedy over the blind roster answers it when both exist.
#
# ONE MOVED FLAG against `labelwise_dense_s2026`: the fit. `--fold -1 --ema` in place of fold 0.
# Pooling, slices, band 0.1-0.9, rate, epochs, `--calibrated-targets` and `--hold-annotations` all held.
# `--hold-annotations` is kept deliberately: `knee-9198` forbids RETRAINING THE ROSTER under it, not
# carrying it in a refit of one member that already had it, which is the same thing
# `calib_dense_hold_ff` does.
#
# NO HOLDOUT NUMBER WILL EXIST FOR IT, BY CONSTRUCTION — see `merged_span_ff.sh`. The parent's 0.8613 is
# the fold-0 number and belongs to the fold-0 member.
#
# HARNESS CHECK. The printed recipe must read `"fold": "-1"`, `"ema": "True"`, `"slices": "20"`,
# band [0.1, 0.9], `calibrated_targets true` and `hold_annotations true`, and the log must name
# `labelwise` as the pooling.
#
# deliberate: --fold
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

FIT="--fold -1 --ema"

run_train labelwise_dense_ff \
  $RECIPE $FIT --epochs 12 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --labels "$LABELS" --out "$MODELS/labelwise_dense_ff.pt"
