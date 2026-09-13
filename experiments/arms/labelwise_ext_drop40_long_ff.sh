#!/usr/bin/env bash
# bead: knee-sexix
# unmeasurable: labelwise_ext_drop40_long_ff fits --fold -1, so it holds no study back and no ledger row
#               can price it (knee-s8kim). Its only ruler is the board row of the entry it feeds, which
#               costs a slot. What licenses the spend is that its holdout TWIN is measured on 909 rows
#               at 0.9045 ledger75 and returns CONVERGED under convergence_check.
#
# WHAT THIS ARM IS
#   `labelwise_ext_drop40_long`'s byte-identical line plus `--fold -1 --ema`. Nothing else moves.
#
# THE FACT THAT STARTED THIS
#   `comp/champion.json`'s six members are final_ws_cf, calib_span_ff, grade_ladder_cf,
#   stack_increments_ff, lr4e-5_ws_cf, through_dense_cf. NOT ONE OF THEM HAS A holdout.npz -- they are
#   _cf/_ff members, fitted at --fold -1, so each saw every study. Our entire honest pool holds 909
#   studies back. The champion's seats are trained on ~4400 studies and every member this project has
#   been ranking is trained on ~3500.
#
# WHY THAT MATTERS FOR THE SEAT
#   `labelwise_ext_drop40_long_s2026` is the best honest solo the project owns (0.9045 ledger75,
#   knee-cgt9m, +0.0059 over its control). It cannot be dropped into the champion as a like-for-like
#   upgrade: it would be the only seat in the entry that never saw a fifth of the corpus. Comparing our
#   honest members against each other says nothing about what a full-corpus refit of the same recipe is
#   worth in the entry, and that comparison is what every seat decision has rested on.
#
# WHY IT IS THE BEST CARD AVAILABLE
#   knee-0v8yz: only MEMBER QUALITY at count 1 has ever moved this board 0.005+. Seating is +0.0017
#   out-of-fold (bf880d89), the CNN family is -0.0038, ten decision rules span 0.0010. This arm is the
#   one operation that puts a better member into a seat the champion already holds, and it adds ~25%
#   more training data on top of a recipe that just cleared its band. Owner directive, standing:
#   "i would be glad if you could use more data - ive always been all for that and dont know why it
#   never happened." This is why it never happened -- the honest harness cannot score a full-fold
#   member, so nothing in the offline loop ever asks for one.
#
# NOT AN EINSTEIN REPEAT
#   `labelwise_dense_ff` is the only full-fold labelwise arm on disk and it is 12 epochs, no
#   `--scheme extended`, no `--slot-dropout`. The two operators that moved the number are exactly the
#   ones it lacks.
#
# PRE-REGISTRATION (board, since there is no offline ruler)
#   Ship as a seat swap against the champion's weakest member, or as the champion seven plus this seat,
#   and say in the submission message which.
#   >= +0.005 board   the knee-0v8yz axis fires and member quality at one seat is the lever.
#   0.944 flat        a better member at one seat does not move a seven-seat entry, which is the
#                     strongest evidence yet that the 0.944 ceiling is the ENTRY SHAPE and not the pool
#                     (knee-fix1t).
#
# deliberate: --fold -1 --ema — the champion's own fit, matched so the new seat is shaped like the ones
#             beside it rather than like our honest pool.
# deliberate: --slot-dropout 0.40 — imported operating point (knee-yotj8), not a swept knob.
# deliberate: --epochs 24 --scheme extended --slices 20 --batch 4 --accumulate 3 --calibrated-targets
#             --hold-annotations — inherited verbatim from the measured twin. This arm adds two flags.
#
# CARD. Same footprint as the twin, which ran on this card. ~25% more studies per epoch, so ~2.9 h.
#   Nothing else of ours may share the card (`card_check`).
#
# control  labelwise_ext_drop40_long_s2026 (on disk, identical line at fold 0). No control run is owed,
#          and no offline comparison between the two is meaningful — the full-fold member has seen the
#          909 rows the other is scored on.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

FIT="--fold -1 --ema"

run_train labelwise_ext_drop40_long_ff \
  $RECIPE $FIT --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --scheme extended \
  --calibrated-targets --hold-annotations \
  --batch 4 --accumulate 3 --slot-dropout 0.40 \
  --labels "$LABELS" --out "$MODELS/labelwise_ext_drop40_long_ff.pt"
