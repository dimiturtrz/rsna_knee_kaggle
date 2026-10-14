#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/blend_ws.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/blend_calib.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/blend_ladder.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/blend_through.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/refit/blend_base.toml "$@"
