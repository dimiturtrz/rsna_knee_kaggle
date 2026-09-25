#!/usr/bin/env bash
# labelwise_depth24_distill.sh — the shipping line plus fold-0-clean own-OOF distillation, one variable.
#
# bead: knee-bgvmd
# reading: labelwise_depth24_distill_s2026.holdout.npz
#
# The term read +0.0056 on the 12-epoch max line (`seat_clean_distill` vs `band_dense`) and never ran under
# labelwise pooling. `teacher_clean.npz` excludes fold 0, so the holdout stays unleaked (knee-zwcyk).

source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"
TEACHER="$PROCESSED/teacher_clean.npz"
require "$TEACHER"

run_train labelwise_depth24_distill \
  $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 24 --batch 10 --vram-gb 22 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --hold-annotations --slot-dropout 0.40 \
  --distill-weight 0.7 --teacher-targets "$TEACHER" \
  --labels "$LABELS" --out "$MODELS/labelwise_depth24_distill_s2026.pt"
