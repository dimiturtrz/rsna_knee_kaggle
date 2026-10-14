#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/calib_span_ff.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/grade_ladder_ff.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/through_dense_ff.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/lr4e-5_ws_ff.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/base_dense_ff.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/band_dense_long_ff.toml "$@"
