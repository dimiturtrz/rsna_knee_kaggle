#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/final_ws_cf.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/grade_ladder_cf.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/through_dense_cf.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/lr4e-5_ws_cf.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/merged_blend_cf.toml "$@"
