#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/basinsoup/basinsoup2_c201.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/basinsoup/basinsoup2_c202.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/basinsoup/basinsoup2_c203.toml "$@"
