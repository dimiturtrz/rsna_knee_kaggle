#!/usr/bin/env bash
# native224_long.sh — native224's exact recipe on a 24-epoch budget instead of 12. The CONTROL half of
# a matched pair with `mixup224_long.sh`, and on its own the cheap gate that decides whether the
# regulariser lane deserves any more card at all.
#
# bead: knee-x580
#
# WHY THIS ARM EXISTS: THE DECLINE HAS NEVER BEEN OBSERVED AT 224. Every number in the overfitting
# account was measured on the 336/dense cell — `band_dense_long` peaked at epoch 6 of 24 and slid
# -0.0182 to the end, `band_span_hold` repeated the shape, `aug_long` and `half_data` both perturbed it
# and read -0.0169 against that -0.0182. At 224 we have only twelve epochs, and over them `native224`
# peaks at 9 and slides -0.0016 to epoch 11 — a quarter of the floor, which is to say nothing. So the
# phenomenon the whole lane is aimed at is, at this resolution, UNMEASURED. Tonight's mixup pair was
# run against a control that had not been shown to have the defect mixup is supposed to cure.
#
# WHAT TONIGHT'S PAIR ACTUALLY SHOWED, and why it points here rather than at a third concentration.
# mixup at 0.2 read 0.8496 and at 0.4 read 0.8516 against 0.8502 — three numbers inside 0.002, a third
# of the 0.0062 floor, with no ordering. The blend demonstrably fired: final train loss 0.6066 control,
# 0.7369 at 0.2, 0.7908 at 0.4, monotone in the knob. So the regulariser did real work to the fit and
# the holdout did not move. That is the same signature `aug_long` produced, and `half_data` before it.
#
# THREE PERTURBATIONS, ONE INVARIANT. Halving the training data left the decline at -0.0169. Turning on
# the intensity block left it at -0.0169. Blending examples and their targets moved the holdout by less
# than a third of the floor. A quantity that survives halving the dataset, changing the appearance, and
# convexly mixing the (study, label) pair is not plausibly memorisation of that pair. The diagnosis this
# project has carried for a week is outvoted by its own three tests, and the suspect it never examined
# is the SCHEDULE: one_cycle's anneal over 24 epochs, where the peak-at-6 and the slide were both
# measured, and where the LR is doing something quite different at epoch 6 than at epoch 23.
#
# WHY THE CONTROL RUNS BEFORE ANY MORE MIXUP. If 24 epochs at 224 reproduce a ~0.018 slide, the defect
# is real at this resolution, the 12-epoch pair was simply too short to see it, and `mixup224_long` is
# worth its own forty minutes. If 24 epochs at 224 do NOT slide, then the decline was a property of the
# dense/336 cell and the regulariser lane has been aimed at a defect that does not exist where we are
# now running. That second outcome closes an axis and is worth as much as the first.
#
# PRE-REGISTERED READINGS, written before the run:
#   DEFECT REAL   peak-to-finish decline >= 0.0124 (2x floor), matching the dense cell's -0.0182 in
#                 kind. Licenses mixup224_long as a matched cure, and re-opens the memorisation account
#                 only if the cure works where the 12-epoch version did not.
#   NO DEFECT     decline <= 0.0062 (the floor). The 224 cell does not overfit on this budget, the
#                 mixup pair was a cure without a disease, and the lane CLOSES here rather than at a
#                 third concentration. The suspect moves to the dense cell's geometry, not to data.
#   AMBIGUOUS     decline between the floor and 2x it. Read the mixup arm's decline beside it and say
#                 so plainly; do not promote a sub-2x slide into a mechanism.
#   Height is NOT the reading. A 24-epoch peak above 0.8502 would be welcome but is not what this arm
#   is for, and a peak on the LAST epoch means the budget bound the run (convergence_check UNDERTRAINED)
#   and the decline question was not answered at all.
#
# control  native224 = 0.8502 over 12 epochs. This arm changes ONE key, --epochs 12 -> 24. Same 224 px,
#          same 12 slices, same 0.20-0.80 band, whole-stack, 4e-5, same labels, same 909-study holdout.
# cost     ~38 min at 94 s an epoch alone; longer sharing the card with mixup224_long, which is
#          deliberate — the pair is the unit of evidence and both arms stretch equally.
#
# HARNESS CHECK BEFORE READING ANY NUMBER. The printed recipe must say "image_size": "224",
# "epochs": "24" and NO mixup, and the log must load 4,406 studies and hold out 909.
#
# deliberate: --epochs 24 (the arm's thesis)
# deliberate: --image-size 224 (native224's control, per that row's standing instruction)
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train native224_long \
  $RECIPE --epochs 24 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --image-size 224 \
  --labels "$LABELS" --out "$PROCESSED/native224_long_s2026.pt"
