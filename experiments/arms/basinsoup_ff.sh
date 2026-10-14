#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/basinsoup/basinsoup_ff_c301.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/basinsoup/basinsoup_ff_c302.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/basinsoup/basinsoup_ff_c303.toml "$@"
