#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/grand_refit/grand_through_dense.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/grand_refit/grand_through_dense_ff.toml "$@"
