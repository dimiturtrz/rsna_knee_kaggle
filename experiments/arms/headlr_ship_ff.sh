#!/usr/bin/env bash
# headlr_ship_ff.sh — headlr_ship's own line, refit on every study.
#
# bead: knee-33iwi
# unmeasurable: this arm fits --fold -1, so it holds no study back and no ledger row can price it
#               (knee-s8kim). Its holdout TWIN headlr_ship_s2026 reads 0.9059 ledger75 / 0.8702
#               midpoint, peak e17 of 24 — the highest offline75 member on disk.
#
# WHAT THIS ARM IS. `headlr_ship.sh`'s flag line plus `--fold -1 --ema`, the matched refit line of
# `labelwise_depth24_drop40_long_ff` and `labelwise_ext_drop40_long_ff`. One variable against the twin.
#
# WHY REFIT AND NOT A KNOB. Refitting is the only positive measured BOARD term this repo owns: +0.008
# on labelwise_ext_drop40_long and +0.003 on labelwise_depth24_drop40_long, each against its own
# measured half-fit twin. The term is per-member and its SIZE may not be predicted (knee-22nk5), but
# both readings are positive and the larger exceeds the whole ensemble premium of +0.001..+0.002.
#
# PRE-REGISTRATION (board, solo entry). >= 0.947: this member becomes the shipping solo and the refit
#   road carries to the highest point we own. 0.944-0.946: refit reproduces a third time at the top and
#   the solo line is level with the champion ensemble. <= 0.943: the refit term does not survive at the
#   top of the offline75 range, and the road stops here rather than at a fourth member.
#
# deliberate: --fold -1 --ema — the refit line, matched to both prior _ff arms.
# deliberate: --head-lr 2e-4 --slot-dropout 0.40 --epochs 24 --slices 24 --batch 10 — the twin's, byte
#             for byte. --batch is what keeps RECIPE's --rank-weight pairwise population matched.
# deliberate: --vram-gb 22 — owner VRAM cap 50-70% of the card.
#
# CARD. ~25% more studies per epoch than the twin's 221 s, ~3 h. Nothing else of ours on the card.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

FIT="--fold -1 --ema"

run_train headlr_ship_ff \
  $RECIPE $FIT --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 24 --batch 10 --vram-gb 22 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --hold-annotations --slot-dropout 0.40 \
  --head-lr 2e-4 \
  --labels "$LABELS" --out "$MODELS/headlr_ship_ff.pt"
