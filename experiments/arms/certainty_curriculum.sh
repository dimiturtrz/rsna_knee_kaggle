#!/usr/bin/env bash
# certainty_curriculum.sh — queue Q11 (knee-6t4l): certainty ramp vs. its uniform control.
#
# The dense recipe is the disagreed_sampler control's exactly: $RECIPE, 12 epochs, whole-stack,
# 20 slices, the 0.1-0.9 band, rank-weight 1.0. The arm adds --certainty-ramp 6; the control draws
# uniformly via --sample-spread ... --sample-alpha 0.0. If disagreed_uniform_s2026 already exists,
# it IS the control and only the ramp arm is queued.
#
# Run from the repo root or anywhere; the script resolves its own location.

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/experiments/lib.sh"
require "$LABELS"

SPREAD="$PROCESSED/train_spread_honest12.npz"
# The spread file is only needed for the explicit control. The ramp arm does not use it.

RAMP_OUT="$MODELS/certainty_ramp_s2026.pt"
CONTROL_OUT="$MODELS/disagreed_uniform_s2026.pt"

# The arm: full dense recipe with the certainty curriculum.
run_train certainty_ramp \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --certainty-ramp 6 \
  --labels "$LABELS" --out "$RAMP_OUT"

# The control: only if the disagreed_sampler uniform control has not already been queued/run.
# A completed checkpoint is the strongest evidence that the control exists; the .metrics.jsonl
# recipe row alone can be written by a failed launch, so we require the member file.
if [ -f "$CONTROL_OUT" ]; then
  echo "control exists: $CONTROL_OUT — skipping the explicit control run"
else
  run_train certainty_uniform \
    $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
    --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
    --sample-spread "$SPREAD" --sample-alpha 0.0 \
    --labels "$LABELS" --out "$CONTROL_OUT"
fi
