#!/usr/bin/env bash
# recal_warm_dense.sh — does labels v3 compose with warm_calib_dense? ONE moving token.
#
# THE ONE MOVING TOKEN is `--labels` (v2 -> `report_labels_v3.csv`). Everything else is
# `warm_calib_dense`'s, copied from `experiments/arms/warm_calib_dense.sh` line for line: same epochs, same
# pooling, same backbone-lr, same slices, same band, same flags. `--eval-labels` is PINNED to v2 and
# is NOT a second knob — it is the ruler refusing to move with the arm. `recal_holdannot` left it
# unset, was scored against the RE-MAPPED table, and its +0.0477 was two rulers rather than a result.
#
# WHY THIS ARM EXISTS. v3 is the one priced knob no W8 seat carries: +0.0045 (band_dense_hold ->
# recal_dense) and +0.0052 (lr4e-5_ws -> recal_holdannot), neither pair carrying calib. `recal_calib`
# prices the composition with calibration first; this arm and its five siblings are what that branch
# buys. Seats are armed in W8-MASS order (1.6269, rank 2 of 6) so the card spends its first
# hours on the column mass that can actually move the block.
#
# WHY IT COULD FAIL — the mechanism to write down if it does. v3's fitted levels and this recipe's own
# target treatment can be two answers to the SAME question (what positive rate does this reader value
# mean). If so the second has nothing left to correct and the two may over-correct together.
#
# PRE-REGISTERED against the control below, own holdout, L75mid, n=909.
#   >= 0.8961  COMPOSES — the seat is re-fit on v3 and the roster line moves.
#   0.8918-0.8961  PARTIAL — ledger it as a half-knob; the seat keeps v2.
#   <= 0.8916  DOES NOT COMPOSE — ledger the two-answers-to-one-question mechanism; seat keeps v2.
#
# deliberate: --labels (the single moving token vs warm_calib_dense)
# deliberate: --eval-labels (the ruler must not move with the arm; recal_holdannot is the incident)
# bead: knee-7yug3
# reading: recal_warm_dense_s2026.holdout.npz
# control  warm_calib_dense_s2026 = 0.8916 (own holdout, L75mid, n=909)
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

RECAL="$PROCESSED/labels/report_labels_v3.csv"
require "$RECAL"
require "$LABELS"

run_train recal_warm_dense \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT  \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations  \
  --encoder-weights "$MODELS/encoders/calib_dense_hold.enc.pt"  \
  --labels "$RECAL" --eval-labels "$LABELS" --out "$MODELS/recal_warm_dense_s2026.pt"
