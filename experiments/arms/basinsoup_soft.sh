#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/basinsoup/basinsoup_soft_c701.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/basinsoup/basinsoup_soft_c702.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/basinsoup/basinsoup_soft_c703.toml "$@"
