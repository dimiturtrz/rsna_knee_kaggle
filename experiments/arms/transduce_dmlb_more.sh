#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/targets/transduce/transduce_dmlb_c409.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/targets/transduce/transduce_dmlb_c410.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/targets/transduce/transduce_dmlb_c411.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/targets/transduce/transduce_dmlb_c412.toml "$@"
