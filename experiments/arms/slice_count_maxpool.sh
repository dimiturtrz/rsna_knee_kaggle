#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/geometry/slicecount/maxpool_12.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/geometry/slicecount/maxpool_20.toml "$@"
