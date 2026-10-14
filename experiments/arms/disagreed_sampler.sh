#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/targets/sampler/disagreed_uniform.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/targets/sampler/disagreed_full.toml "$@"
