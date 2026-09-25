#!/usr/bin/env bash
# bead: knee-6u1tq
# unmeasurable: shift4_depth24_ff fits --fold -1, so it holds no study back and no ledger row can price it
#               (knee-s8kim). Its holdout TWIN shift4_depth24_s2026 is the reading; this arm is armed only
#               once that twin beats labelwise_depth24_drop40_long_s2026's 0.9053 ledger75.
#
# WHAT THIS ARM IS
#   `labelwise_depth24_drop40_long_ff` plus `--slice-shift 4`: the full-data refit of the shipping line with
#   the one flag its fold-0 twin measures. Full data read about +0.003 board over the fold-0 solo on this
#   line (l75solo 0.941 vs ffsolo 0.944), which is the only route from a fold-0 win to a slot.
#
# PRE-REGISTRATION (board, solo). >= 0.947 beats the champion's 0.946 and becomes the entry.
#   0.946 ties at the board's 0.001 resolution: parked as a seat candidate. <= 0.945: parked.
#
# deliberate: every flag is labelwise_depth24_drop40_long_ff's byte for byte; only --slice-shift 4 moves.
#
# CARD. ~25% more studies per epoch than the twin's measured pace; nothing else of ours on the card.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

FIT="--fold -1 --ema"

run_train shift4_depth24_ff \
  $RECIPE $FIT --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 24 --batch 10 --vram-gb 22 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --hold-annotations --slot-dropout 0.40 --slice-shift 4 \
  --labels "$LABELS" --out "$MODELS/shift4_depth24_ff.pt"
