#!/usr/bin/env bash
# llrd224_flat.sh — the SAME ladder as `llrd224.sh` (`--layer-decay 0.75` over the default six opened
# blocks) with the top rate left at the control's 4e-5, i.e. WITHOUT the mean-rate compensation. The
# second point of the pair, and the one that makes the first point readable.
#
# bead: knee-s9cr
#
# THIS ARM IS AN INSTRUMENT AND IS PRICED AS ONE. It is not expected to beat the control; its own
# pre-registration below predicts it will lose. It earns nineteen minutes because it is the only thing
# that turns `llrd224`'s number into a statement about the ladder SHAPE rather than about the mean rate,
# and because it settles, at a third of the cost, the question the ledger left open when it refused
# `band_dense_llrd`'s verdict (README:2074-2115).
#
# THE THREE-POINT DESIGN, and what each pair separates. Control `native224` = flat 4e-5 over six blocks.
# `llrd224` = graded, mean rate 4e-5 (top 8.2e-5, bottom 1.95e-5). This arm = graded, TOP rate 4e-5, mean
# 1.95e-5 (the factors 1, 0.75, 0.5625, 0.4219, 0.3164, 0.2373 have geometric mean 0.75^2.5 = 0.487, and
# `EncoderSurgery.backbone_groups`, `kneemri/models/encoder_surgery.py:236-244`, anchors the ladder at
# the TOP block so grading can only ever slow what is below it). Control vs this arm is the naive LLRD
# comparison — the one `band_dense_llrd` ran, and the one that cannot distinguish a depth effect from a
# rate cut. This arm vs `llrd224` holds the ladder shape fixed and moves only the mean rate. `llrd224`
# vs control holds the mean rate fixed and moves only the shape. Three points, two clean contrasts, one
# confound removed.
#
# WHY THE ANSWER IS NOT ALREADY KNOWN FROM THE RATE SWEEP. It nearly is, and saying so is the honest
# price of this arm: `rate_isolate` measured 8e-6 at 0.8323 against 4e-5's 0.8525, so a mean rate of
# 1.95e-5 sits between a point that clearly underfits and the control, and the expected reading is a
# loss of roughly half that gap. What the sweep CANNOT supply is the same number under a graded ladder
# — if grading is worth something, this arm's loss is smaller than the rate cut alone predicts, and that
# is a second, independent read on the same shape question `llrd224` asks from the other side.
#
# control  native224 = 0.8502, best at epoch 7, converged not truncated. Same twelve epochs, twelve
#          slices, 0.20-0.80 band, whole-stack, 224 px, six opened blocks, head 1e-3, same holdout. The
#          single variable against the CONTROL is --layer-decay; the single variable against `llrd224`
#          is --backbone-lr.
# cost     ~19 min for twelve epochs at 94 s, ~8124 MiB reserved.
# price    19 min for P(clears the floor) = 0.05, and that is the point — a pre-registered near-null run
#          justified as an INSTRUMENT, not sold as an improvement. It retires the suspicion that
#          `band_dense_llrd`'s −0.0095 was a depth result, and it licenses reading `llrd224` as a shape
#          measurement. Without it, `llrd224` is one more single number inside or outside a floor with a
#          second variable nobody priced, which is the exact failure this pair exists to not repeat.
#
# HARNESS CHECK BEFORE READING ANY NUMBER. The printed recipe must say "layer_decay": 0.75,
# "backbone_lr": 4e-05 and "image_size": "224". `Trainer.encoder_rate_decay` (`trainer.py:166`) reads the
# applied ratio off the optimiser's top two backbone groups, so the decay cannot be silently absent.
#
# THE FALSIFIABLE PART, written before the run, READ AS A PAIR with llrd224.
#   EXPECTED    the peak lands between 0.8400 and 0.8500 — a loss of roughly half the 8e-6-to-4e-5 gap,
#               which is what a 2.05x mean-rate cut predicts with no depth effect at all. This is the
#               modal branch and it CONFIRMS the instrument rather than the method: it says the prior
#               arm's −0.0095 was a rate cut, `band_dense_llrd`'s refused verdict is now settled as
#               refused, and `llrd224` is the arm to read.
#   GRADING HELPS   the peak lands ABOVE 0.8500 despite the halved mean rate. Grading is then buying
#               back a rate cut, which is a stronger statement than `llrd224` clearing its floor, and
#               the two together would make the depth axis the best-supported open lever on the board.
#   GRADING HURTS   the peak falls below 0.8400, past what the rate cut alone accounts for. Depth
#               grading is actively wrong on this trunk and the axis closes for real — the closure
#               `band_dense_llrd` claimed and was not entitled to.
#   NO BRANCH HERE IS A SUBMISSION. This arm produces a member only incidentally; nothing about its
#               score is expected to earn a seat and it is not to be sent to `member_curve` on the
#               strength of a number that its own header predicted would be low.
#
# deliberate: --layer-decay 0.75 (the same ladder as llrd224, uncompensated — the arm's thesis)
# deliberate: --image-size 224 (native224's control, per that row's standing instruction)
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train llrd224_flat \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --image-size 224 --layer-decay 0.75 \
  --labels "$LABELS" --out "$PROCESSED/llrd224_flat_s2026.pt"
