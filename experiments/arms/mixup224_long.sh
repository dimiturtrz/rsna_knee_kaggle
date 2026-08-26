#!/usr/bin/env bash
# mixup224_long.sh — `--mixup 0.4` on the 24-epoch budget. The CURE half of a matched pair with
# `native224_long.sh`, which is the control and the gate.
#
# bead: knee-x580
#
# WHY 0.4 AND NOT 0.2. Tonight's dose pair read 0.8496 at 0.2 and 0.8516 at 0.4 against a 0.8502
# control — all three inside 0.002, so neither concentration wins on HEIGHT and the ordering is not
# signal. What separates them is SHAPE: the control peaks at epoch 9 and turns down, 0.2 peaks at 10
# and plateaus, 0.4 peaks on its LAST epoch and never turns down at all (0.8510, 0.8514, 0.8515, 0.8516
# across epochs 8-11). Final train loss is monotone in the knob — 0.6066, 0.7369, 0.7908 — so the blend
# fired harder at 0.4 and the fit was still improving when the budget ended. If the thesis is that
# mixup buys a LONGER usable schedule, the concentration to test it at is the one whose curve had not
# finished, and 0.4 is that one. Running 0.2 here would spend forty minutes on the arm that had already
# flattened.
#
# THIS ARM IS GATED ON ITS OWN CONTROL AND SHOULD BE READ SECOND. `native224_long` decides whether the
# defect mixup is supposed to cure exists at 224 at all. Every number in the overfitting account —
# band_dense_long's -0.0182 from epoch 6 to 23, aug_long's -0.0169, half_data's -0.0169 — was measured
# on the 336/dense cell. At 224 we have twelve epochs and a -0.0016 slide, which is a quarter of the
# floor. So this arm can only be a cure if the control shows a disease.
#
# WHAT WOULD MAKE THIS ARM A RESULT RATHER THAN A FOURTH NULL. Three perturbations aimed at
# memorisation of the (study, label) pair have now left the decline invariant: halving the data
# (-0.0169), intensity augmentation (-0.0169), and blending at two concentrations (holdout flat inside a
# third of the floor). The honest prior after that is that the decline is NOT about the data, and this
# arm is the last cheap test of the data account before the suspect moves to the SCHEDULE — one_cycle's
# anneal, which is doing something quite different at epoch 6 than at epoch 23 and has never been
# varied.
#
# PRE-REGISTERED READINGS, written before the run, and read only after the control:
#   CURE          the control declines >= 2x floor and this arm's decline is smaller than the control's
#                 by more than the floor. That is the first time any regulariser has touched this
#                 quantity, and it re-opens the memorisation account that the three nulls closed.
#   NO DISEASE    the control does not decline. This arm is then uninterpretable as a cure, and its
#                 only reading is its height. Say that plainly rather than mining the curve.
#   FOURTH NULL   the control declines and this arm declines by the same amount within the floor. The
#                 data account is then refuted on four independent perturbations and the lane closes;
#                 the schedule becomes the named next suspect. This is the outcome the prior favours.
#   Height above 0.8502 is welcome and is not the reading. A peak on the LAST epoch means the budget
#   bound the run and the decline question was not answered.
#
# CONFOUND, NAMED BEFORE THE RUN, unchanged from mixup224. --report-weight is off here and must stay
# off: ReportSupervision is handed the batch's POSITIONS rather than a table of rows, so it alone would
# read un-blended under --mixup. The label, ranking and teacher terms all read blended rows. See
# kneemri/training/mixup.py.
#
# control  native224_long, running beside this arm on the same card. The single variable is --mixup.
# cost     ~38 min at 94 s an epoch alone; longer sharing the card with its control, which is
#          deliberate — both arms stretch equally and only their scores are read.
#
# HARNESS CHECK BEFORE READING ANY NUMBER. The printed recipe must say "image_size": "224",
# "epochs": "24" and "mixup": 0.4, and the log must load 4,406 studies and hold out 909.
#
# deliberate: --mixup 0.4 (the arm's thesis)
# deliberate: --epochs 24 (the budget the decline was measured over)
# deliberate: --image-size 224 (native224's control, per that row's standing instruction)
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train mixup224_long \
  $RECIPE --epochs 24 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --image-size 224 --mixup 0.4 \
  --labels "$LABELS" --out "$PROCESSED/mixup224_long_s2026.pt"
