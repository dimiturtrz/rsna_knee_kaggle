#!/usr/bin/env bash
# resnet50d_lr.sh — the CNN lane's SECOND draw, and the only thing that changes is the backbone rate.
#
# WHY A SECOND ARM AT ALL. The family order-statistic null (D117/D118) rejects for five non-`default`
# families and does NOT reject here: ResNet50d sits at n=1, p=0.27, and it BEAT its own n=1 expectation
# (0.8659 against 0.8614). Every other row in that table is "tried and worse"; this is the one row where
# "not tried" is the true sentence. A family cannot be closed on a single draw, and this is the cheapest
# second draw the project can buy.
#
# WHAT IT COSTS, SAID BEFORE IT RUNS. `resnet50d_short_s2026.metrics.jsonl` records 85-87 s an epoch for
# twelve epochs — SEVENTEEN MINUTES of card, not the ~1 h `resnet50d_short.sh`'s own header guessed. That
# is the entire argument for spending a slot of card time on a lane nobody expects to win: at seventeen
# minutes the expected-gain/wall-clock ratio survives an honest prior of near-zero.
#
# WHY THE RATE AND NOT THE EPOCHS. The short arm's last two epochs read auc 0.83267 then 0.83279 — flat,
# converged, not truncated and not budget-bound (`convergence_check` is silent on it). So the twelve
# epochs are not the binding constraint and lengthening them buys a flat line. The rate is the knob that
# was never swept: 4e-4 was taken from `radnet_family` because it is a CNN, in one step, with no second
# point beside it.
#
# WHY 1.26e-4 AND NOT A ROUND NUMBER. The project holds exactly two fitted backbone rates: 4e-4, the CNN
# rate this family already used, and 4e-5, the ViT corner's. 1.26e-4 is their geometric mean — one step of
# a sqrt(10) sweep, placed between the two rates the repo has actually measured rather than picked for
# looking tidy. A third point either side of it is what a THIRD arm buys, and only if this one moves.
#
# PRE-REGISTERED, before it runs:
#   >= 0.88        the rate was the binding constraint; the family re-opens and a 336px arm queues
#   0.86 - 0.88    the rate is not the constraint; the family is closed ON TWO DRAWS, which is the
#                  sentence D118 asked for and a different one from "closed on n=1"
#   < 0.86         lowering the rate hurt; 4e-4 is at or past the optimum and the sweep's next point is
#                  ABOVE it, not below. That is a finding about the lane, not a null.
#
# bead: knee-mv093
# reading: D:/models/rsna_knee_abnormality/resnet50d_lr_s2026.holdout.npz
#
# deliberate: --backbone-lr 1.26e-4 (the swept knob, the only difference from resnet50d_short)
# deliberate: --image-size 224 (the revision's own resolution; changing it would move two things)
# deliberate: --epochs 12 (matched to the first draw, and the curve is flat there)
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train resnet50d_lr \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack --through-plane $THROUGHPUT \
  --variant resnet50d --image-size 224 --backbone-lr 1.26e-4 \
  --slices 20 --band-low 0.1 --band-high 0.9 \
  --labels "$LABELS" --out "$MODELS/resnet50d_lr_s2026.pt"
