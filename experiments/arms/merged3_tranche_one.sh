#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/merged_blend_dm.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/calib_span_dm.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/lr4e-5_ws_dm.toml "$@"
