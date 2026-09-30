#!/usr/bin/env bash
# bead: knee-p0s4f
# reading: lw24_targetblend_s2026.holdout.npz
#
# WHAT THIS ARM IS
#   `labelwise_depth24_drop40_long` with ONE variable moved: the TRAINING target. `--labels` is
#   `blend_oof_a07_cal.csv` = 0.3 * calibrated(report_labels_v2) + 0.7 * teacher_clean on the 3497 train rows
#   (`python -m kneemri.training.pseudo_table --teacher ... --weight 0.7 --confidence 1.0`); the 909
#   fold-0 holdout rows are the report's own, untouched. `--eval-labels` pins the ruler to
#   report_labels_v2, so the score is against readings, never against the teacher.
#
# WHY — the 735304 thread's one convergent recipe, in its TARGET form. Ours only ever ran distillation
#   as a ranking term beside an untouched target (terms.py:354). hisrecipe_distill ran the target form
#   on a foreign trunk, never on this line and never on the board.
#
# PRE-REGISTRATION. ledger75 is EXPECTED null-or-negative by construction: a report-derived ruler cannot
#   see a move toward annotation truth. It does not decide. The BOARD decides, one INSTRUMENT slot on
#   the solo, bands against basinsoup_ff_k4_s2026 alone (board 0.946):
#   >= 0.949     keep; compose into the soup line.
#   0.945-0.948  tie; the form transfers nothing the soup lacks.
#   < 0.945      the form does not transfer on the board either.
#
# --calibrated-targets remaps exact reader levels only: blended cells pass through (already
# calibrated by pseudo_table --calibrated), holdout raw levels calibrate as in the control.
# deliberate: --labels blend_oof_a07_cal.csv --eval-labels report_labels_v2 — the one variable.
# deliberate: every other flag verbatim from labelwise_depth24_drop40_long.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"
BLEND="$(dirname "$LABELS")/blend_oof_a07_cal.csv"
require "$BLEND"

run_train lw24_targetblend \
  $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 24 --batch 10 --vram-gb 28 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --hold-annotations --slot-dropout 0.40 \
  --labels "$BLEND" --eval-labels "$LABELS" --out "$MODELS/lw24_targetblend_s2026.pt"
