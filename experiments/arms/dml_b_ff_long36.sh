#!/usr/bin/env bash
# bead: knee-bxp8h
# unmeasurable: dml_b_ff_long36 fits --fold -1, so it holds no study back and no ledger row can price it
#               (knee-s8kim). Its only ruler is the board row of its solo entry.
#
# WHAT THIS ARM IS
#   `dml_b_ff` (knee-8bmi2) with `--epochs 36`, the single variable of `depth24_ff_long36` (knee-lz8pd).
#   Both components read on the board tonight; this is the entry that would carry them together.
#
# deliberate: --epochs 36 — depth24_ff_long36's one variable, applied to the peer-supervised line.
# deliberate: --fold -1 --ema --distill-weight 0.3 — dml_b_ff's, unchanged.
# deliberate: --vram-gb 22 — 18 and 20 both OOM on this recipe (dml_b_ff).
#
# CARD. ~4.5 h; the teach file is already on disk, so no teach pass.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"
require "$MODELS/dml_a_s2026.pt"

TEACH="$PROCESSED/dml_read_a_all.npz"
FIT="--fold -1 --ema"

[ -f "$TEACH" ] || run_gpu uv run --no-sync --project "$REPO" python -u -m kneemri.inference.teacher_readings \
  --checkpoints "$MODELS/dml_a_s2026.pt" --out "$TEACH" --teach-all

run_train dml_b_ff_long36 \
  $RECIPE $FIT --epochs 36 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 24 --batch 10 --vram-gb 22 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --hold-annotations --slot-dropout 0.40 \
  --distill-weight 0.3 --teacher-targets "$TEACH" \
  --labels "$LABELS" --out "$MODELS/dml_b_ff_long36.pt"
