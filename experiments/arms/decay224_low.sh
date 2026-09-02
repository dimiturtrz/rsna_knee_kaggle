#!/usr/bin/env bash
# decay224_low.sh — native224's recipe with `--weight-decay 0.005`. The other side of the decay axis,
# and the arm that makes the pair a MEASUREMENT rather than a guess about direction.
#
# bead: knee-s9cr
#
# WHY BOTH DIRECTIONS. 0.02 was inherited beside a backbone rate of 8e-6 and never re-derived after the
# rate sweep moved the trunk 5x to 4e-5 (see `decay224_high.sh` for the constant's history and for the
# flag, which is new code added with these arms). Inheriting a value is not evidence that it is a
# minimum, and it is not evidence that it is too LOW either — a one-sided arm can only say "more is
# better/worse", which leaves the more likely case, that the optimum moved past it in the other
# direction, untested. Two points 4x apart on either side of the inherited value bracket it, and three
# readings settle the axis rather than nudging it.
#
# THE MECHANISM FOR LESS. Decoupled decay shrinks every weight by `lr * wd` per step, so a rate raised
# 5x is already applying five times the shrinkage per step that the constant was chosen under — the
# recipe got a large regularisation increase as a side effect of a rate decision, and nobody priced it.
# On this corpus that could easily be too much: 4,406 studies with twelve noisy report-derived columns
# and 58 image-read labels is a small supervision signal for a 22M-parameter trunk, and the one thing
# this project has measured about capacity is that MORE of it is the only axis that has ever cleared the
# noise floor. Shrinkage is capacity removal. 0.005 is a quarter of the inherited value, which restores
# roughly the per-step shrinkage the recipe had before the rate moved — that is the derivation, and it
# is why the low point is 0.005 rather than a round zero.
#
# NOT ZERO, DELIBERATELY. Zero decay is a different claim (no regularisation at all) and it is the one
# value where AdamW's decoupling stops mattering, so a reading at zero would confound "less decay helps"
# with "the decoupling is doing something odd". A quarter keeps the mechanism intact and still moves the
# coefficient far enough to clear noise.
#
# THE RESOLUTION ARGUMENT, AND HERE I PREDICT LESS EFFECT THAN AT 336. Decay's grip scales with how much
# the weights want to move, and at 224 there is less in-plane structure to fit, so the gradient pull the
# shrinkage competes against is weaker and the same coefficient bites relatively harder — which is the
# argument `decay224_high` makes for MORE regularisation being right here. This arm is the same argument
# read backwards, so an honest prediction is that the LOW point does less at 224 than it would at dense.
# If both this and decay224_high read flat, the axis is insensitive at twelve epochs and the pair has
# closed it; if this one wins, the settled recipe has been over-regularised since the rate sweep landed.
#
# control  native224 = 0.8502, best at epoch 7, converged not truncated. 12 epochs, 12 slices, 0.20-0.80
#          band, whole-stack, backbone 4e-5, 224 px, same labels, same 909-study holdout, decay 0.02.
#          The single variable is --weight-decay. Read against the control first, then against
#          decay224_high — the bracket is what the two arms buy together.
# cost     ~19 min for twelve epochs at 94 s. Same cache as native224; the coefficient is free.
# price    19 min for P(clears the 0.0062 floor on the MEAN) = 0.18, below decay224_high's 0.25: the
#          rate change made the effective shrinkage larger, so the a-priori more likely error is too
#          MUCH decay — but the direction is exactly what is unknown, which is why both points run.
#
# HARNESS CHECK BEFORE READING ANY NUMBER. The printed recipe must say "weight_decay": "0.005" and
# "image_size": "224", 4,406 studies loaded and 909 held out. The landed check reads
# `optimiser_weight_decay` off the built optimiser's first group, not off the flag — decay is an AdamW
# constructor keyword, and a value that never arrives trains the control at 0.02 with 0.005 in its
# config.
#
# THE FALSIFIABLE PART, written before the run.
#   CONFIRMED   the mean peak clears 0.8564 (control + floor). The settled recipe has been
#               over-regularised since the rate sweep, every member in the pool was fitted that way, and
#               the axis wants a third point below 0.005 before anything is shipped on it.
#   DIRECTIONAL positive under the floor. Kept as a free recipe change if decay224_high is not also
#               positive; two positives in opposite directions is noise, not a dose curve, and would
#               mean the axis is flat inside the floor.
#   FLAT        with decay224_high also flat: the decay axis is CLOSED at twelve epochs and 224 px, the
#               inherited constant is exonerated, and the late slide is not weight-norm growth. That is
#               the instrument paying out, and it redirects this family to --grad-clip and --mixup.
#   REFUTED     the mean falls by more than the floor. Less shrinkage is worse, which says the recipe
#               wants MORE and hands the axis to decay224_high; that reading is only usable if this arm
#               and that one are read as one pair, which is the whole point of running both.
#
# deliberate: --weight-decay 0.005 (the arm's thesis — a 5x rate rise already quintupled the shrinkage)
# deliberate: --image-size 224 (native224's control, per that row's standing instruction)
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train decay224_low \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --image-size 224 --weight-decay 0.005 \
  --labels "$LABELS" --out "$MODELS/decay224_low_s2026.pt"
