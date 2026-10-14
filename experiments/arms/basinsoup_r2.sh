#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/basinsoup/basinsoup_r2_c501.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/basinsoup/basinsoup_r2_c502.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/basinsoup/basinsoup_r2_c503.toml "$@"
