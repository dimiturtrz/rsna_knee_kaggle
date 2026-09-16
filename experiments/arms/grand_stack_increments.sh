#!/usr/bin/env bash
# bead: knee-feb62.7
# reading: grand_stack_increments_s2026.holdout.npz
# DRAFT, not armed. Seat stack_increments_ff is --fold -1: the fold-0 twin is the reading; the _ff refit runs only if the twin beats
# the seat's own fold-0 original. max pooling kept, grand's labelwise/hold/slot leaves overridden (grand.toml:20,28,32).
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

# mixup (grand.toml:36) over teacher targets is unverified.
run_train grand_stack_increments \
  --config experiments/recipes/base.toml --config experiments/recipes/grand.toml \
  --set model.slice_pooling=max --set regularisation.slot_dropout=0.0 --distill-weight 0.3 --teacher-targets "$PROCESSED/raptor_teacher.npz" \
  --labels "$PROCESSED/labels/blend_three_v1.csv" --eval-labels "$LABELS" --out "$MODELS/grand_stack_increments_s2026.pt"

run_train grand_stack_increments_ff \
  --config experiments/recipes/base.toml --config experiments/recipes/grand.toml \
  --set model.slice_pooling=max --set regularisation.slot_dropout=0.0 --distill-weight 0.3 --teacher-targets "$PROCESSED/raptor_teacher.npz" --set run.fold=-1 \
  --labels "$PROCESSED/labels/blend_three_v1.csv" --eval-labels "$LABELS" --out "$MODELS/grand_stack_increments_ff.pt"
