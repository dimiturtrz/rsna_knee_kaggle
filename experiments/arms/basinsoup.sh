#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/basinsoup/basinsoup_c101.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/basinsoup/basinsoup_c102.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/basinsoup/basinsoup_c103.toml "$@"
