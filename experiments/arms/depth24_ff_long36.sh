#!/usr/bin/env bash
# bead: knee-lz8pd
# unmeasurable: this arm fits --fold -1, so it holds no study back and no ledger row can price it
#               (knee-s8kim). Its board control is labelwise_depth24_drop40_long_ff at 0.944.
#
# WHAT THIS ARM IS
#   `labelwise_depth24_drop40_long_ff`'s line with `--epochs 36` and nothing else moved. One variable
#   against a control that is already on the board, so it costs no extra slot to interpret.
#
# WHY THIS AND NOT A KNOB
#   The full-corpus refit is the only positive BOARD term this repo owns, +0.0055 and flat across two
#   twins whose offline75 are tied inside the 0.0062 floor. Its one unspent compound is the thing more
#   data licenses: a longer anneal. The honest-split 36-epoch reading is -0.0018, inside the paired
#   floor, so it is no reading rather than a null — and it was taken on 3498 studies, the regime where a
#   longer schedule overfits. 4407 studies loosen that by 26 percent.
#
# PRE-REGISTRATION (board, solo entry). >= 0.947: the anneal length compounds with the refit and the
#   schedule axis re-opens on the full corpus. 0.944-0.946: the longer anneal buys nothing the refit did
#   not already have. <= 0.943: a longer schedule COSTS on the full corpus, which prices every future
#   refit at 24 epochs.
#
# deliberate: --epochs 36 — the single variable.
# deliberate: --fold -1 --ema — the refit line, matched to the control byte for byte.
# deliberate: --slot-dropout 0.40 --slices 24 --batch 10 --calibrated-targets --hold-annotations — the
#             control's, unchanged, so the comparison is the schedule alone.
# deliberate: --vram-gb 22 — owner VRAM cap 50-70% of the card.
#
# CARD. ~4.5 h: the 24-epoch control measured ~3 h over the same 4407 studies.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

FIT="--fold -1 --ema"

run_train depth24_ff_long36 \
  $RECIPE $FIT --epochs 36 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 24 --batch 10 --vram-gb 22 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --hold-annotations --slot-dropout 0.40 \
  --labels "$LABELS" --out "$MODELS/depth24_ff_long36.pt"
