#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/merged_blend_df.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/final_ws_df.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/calib_span_df.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/grade_ladder_df.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/through_dense_df.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/lr4e-5_ws_df.toml "$@"
