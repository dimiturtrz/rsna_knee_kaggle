#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/v3_calib_ff.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/v3_grade_ff.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/v3_through_ff.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/v3_ws_ff.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/v3_attend_ff.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/v3_focal_ff.toml "$@"
