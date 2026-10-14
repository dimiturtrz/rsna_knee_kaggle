#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/final_ws_ramp.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/through_dense_ramp.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/grade_ladder_ramp.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/lr4e-5_ws_ramp.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/calib_span_ramp.toml "$@"
