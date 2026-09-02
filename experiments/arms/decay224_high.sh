#!/usr/bin/env bash
# decay224_high.sh — native224's recipe with `--weight-decay 0.1` and nothing else moved. The first arm
# in this project to move AdamW's decay off the value it was born with.
#
# bead: knee-s9cr
#
# THE CONSTANT AND WHERE IT CAME FROM. `trainer.py` carried `WEIGHT_DECAY = 0.02` as a module constant
# with no flag: every number in this ledger was measured at 0.02, and 0.02 was inherited from the
# reference recipe alongside a backbone rate of 8e-6. The rate sweep has since moved the trunk 5x to
# 4e-5 and the decay did not move with it. The flag is new code, added with this arm — the constant is
# now `Adaptation.weight_decay`, read back off the built optimiser's first parameter group so a value
# that never reached `AdamW` refuses before the card is spent.
#
# THE MECHANISM, AND WHY IT IS A REAL AXIS RATHER THAN A KNOB. Decoupled decay pulls every weight toward
# zero by `lr * wd` per step, so the norm a weight settles at is set by the RATIO of the gradient's pull
# to that shrinkage — and both terms scale with the rate. Raising the rate 5x with the decay fixed keeps
# that ratio unchanged in the linear regime, but our encoder is not in it: at 8e-6 the trunk barely
# moved (that is the measured reason the depth bracket read neutral), and at 4e-5 it moves enough for
# regularisation to have something to act on for the first time. 0.02 was therefore not chosen for the
# recipe that is now running — it was chosen for one in which it could hardly matter.
#
# WHY UP, AND WHY 0.1. Up, because the failure this family is chasing is a run that keeps fitting while
# its holdout falls (`native224_long`, −0.0165 over twelve extra epochs) — that is the shape decay
# exists to fight, and it is the same measurement `clip224` reads a different way. 0.1 because it is the
# value the ViT fine-tuning recipes this trunk descends from use (AdamW's paper, DeiT and MAE all sit at
# 0.05-0.1), it is 5x the inherited value against a rate that moved 5x, and a 5x step is the smallest
# one that can clear this project's noise floor on a regularisation axis — 0.03 against 0.02 would be a
# coin toss dressed as an arm. `decay224_low.sh` takes the other direction at 0.005.
#
# THE RESOLUTION ARGUMENT, AND I PREDICT MORE AT 224 THAN AT 336. Decay's job is to stop the model
# spending capacity on structure that does not generalise. At 224 each token averages ~2.25x more
# tissue, so there is LESS real in-plane structure available and the same encoder capacity sits over a
# weaker signal — the classic condition for over-fitting the noise that remains. The competing reading,
# that a coarser input needs every parameter it can keep and decay just removes capacity, predicts LESS,
# and would show as this arm losing while the same flag helps at dense. Either way this is the first
# reading the project will have of the axis at all.
#
# control  native224 = 0.8502, best at epoch 7, converged not truncated. 12 epochs, 12 slices, 0.20-0.80
#          band, whole-stack, backbone 4e-5, 224 px, same labels, same 909-study holdout, decay 0.02.
#          The single variable is --weight-decay.
# cost     ~19 min for twelve epochs at 94 s. Same cache as native224; decay is free — it is already
#          inside AdamW's update and only the coefficient changes.
# price    19 min for P(clears the 0.0062 floor on the MEAN) = 0.25, the highest-priced arm in this
#          window. Not because regularisation is glamorous, but because this is a value nobody chose:
#          the odds that the inherited constant is the right one for a recipe whose rate has moved 5x
#          are not high, and being WRONG here is worth as much as being right — a loss at 0.1 with a
#          gain at 0.005 says the recipe wants less decay, which is a finding about the settled recipe
#          rather than about this arm.
#
# HARNESS CHECK BEFORE READING ANY NUMBER. The printed recipe must say "weight_decay": "0.1" and
# "image_size": "224", and the log must load 4,406 studies and hold out 909. The landed check reads
# `optimiser_weight_decay` off the built optimiser rather than off the flag, which is the check that
# matters here: decay is a constructor keyword, so a value that never arrives leaves the run training at
# 0.02 with 0.1 in its config — the control's number under this arm's name.
#
# THE FALSIFIABLE PART, written before the run.
#   CONFIRMED   the mean peak clears 0.8564 (control + floor). The settled recipe is under-regularised
#               at its current rate; every member in the pool was fitted that way, and the whole roster
#               becomes a refit candidate. That is the largest consequence any arm in this window has.
#   DIRECTIONAL positive under the floor. Kept and carried into the 336 recipe as a free change, but the
#               claim stays "0.02 was not optimal", not "decay is the lever".
#   FLAT        the axis is insensitive between 0.02 and 0.1 at twelve epochs, which is itself worth the
#               slot: it retires the suspicion that the inherited constant is silently costing us, and
#               it says the late slide is not weight-norm growth. Read with decay224_low before closing.
#   REFUTED     the mean falls by more than the floor. 0.1 is too much shrinkage for a 4,406-study
#               corpus with 58 image-read labels; the axis then wants LESS decay, and decay224_low is
#               where that reading continues rather than a re-test of this one.
#
# deliberate: --weight-decay 0.1 (the arm's thesis — the inherited 0.02 was set beside a 5x lower rate)
# deliberate: --image-size 224 (native224's control, per that row's standing instruction)
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train decay224_high \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --image-size 224 --weight-decay 0.1 \
  --labels "$LABELS" --out "$MODELS/decay224_high_s2026.pt"
