#!/usr/bin/env bash
# bead: knee-feb62.7
# reading: grand_calib_span_s2026.holdout.npz
# DRAFT, not armed. Seat calib_span_ff is --fold -1: the fold-0 twin is the reading; the _ff refit runs only if the twin beats
# the seat's own fold-0 original. max pooling kept, grand's labelwise/hold/slot leaves overridden (grand.toml:20,28,32).
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"


run_train grand_calib_span \
  --config experiments/recipes/base.toml --config experiments/recipes/grand.toml \
  --set model.slice_pooling=max --set targets.hold_annotations=false --set regularisation.slot_dropout=0.0 --set data.slices=12 \
  --labels "$LABELS" --out "$MODELS/grand_calib_span_s2026.pt"

run_train grand_calib_span_ff \
  --config experiments/recipes/base.toml --config experiments/recipes/grand.toml \
  --set model.slice_pooling=max --set targets.hold_annotations=false --set regularisation.slot_dropout=0.0 --set data.slices=12 --set run.fold=-1 \
  --labels "$LABELS" --out "$MODELS/grand_calib_span_ff.pt"
