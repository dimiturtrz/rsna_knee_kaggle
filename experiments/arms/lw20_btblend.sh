#!/usr/bin/env bash
# bead: knee-p844l
# reading: labelwise_dense_drop40_long_btblend_s2026.holdout.npz
#
# WHAT THIS ARM IS
#   `lw20_targetblend` with ONE input changed: the teacher. `blend_bt_a07_cal.csv` = 0.3 * calibrated
#   (report_labels_v2) + 0.7 * teacher_blend on the 3497 train-fold rows (pseudo_table --calibrated,
#   --confidence 1.0, --weight 0.7, as the parent). `teacher_blend.npz` = rank mean of teacher_clean_full
#   (ours, five-fold OOF) and teacher_foreign_corpus (five-fold OOF by its own split), re-valued per column
#   on the mean of the two sorted marginals so cells stay probabilities.
#
# WHY — the parent's teacher was BELOW its student: on the 909 fold-0 rows teacher_clean_full reads
#   0.8775 ledger75 against the student's 0.8891 and the control's 0.9048. teacher_blend reads 0.9061,
#   the first fold-honest teacher at or above the control (teacher_foreign_corpus alone 0.8964).
#
# PRE-REGISTRATION. Arm ONLY if the parent's board read (tblendsolo, slot 2026-10-13) lands >= 0.938:
#   that is the evidence the ruler under-reads a teacher-relabelled student. ledger75 does not decide
#   (report ruler, AXES:163); below control by more than 0.02 means lost signal, do not ship. Otherwise
#   the BOARD decides as a solo kernel against the parent's board read: >= parent + 0.003 the teacher
#   upgrade transfers; inside that, flat.
#
# deliberate: --labels blend_bt_a07_cal.csv --eval-labels report_labels_v2 — the one variable.
# deliberate: every other flag verbatim from lw20_targetblend.
#
# control  labelwise_dense_drop40_long_targetblend_s2026 = 0.8891 ledger75 (on disk). No control run is owed.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"
BLEND="$(dirname "$LABELS")/blend_bt_a07_cal.csv"
require "$BLEND"

run_train labelwise_dense_drop40_long_btblend \
  $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --slot-dropout 0.40 \
  --labels "$BLEND" --eval-labels "$LABELS" \
  --out "$MODELS/labelwise_dense_drop40_long_btblend_s2026.pt"
