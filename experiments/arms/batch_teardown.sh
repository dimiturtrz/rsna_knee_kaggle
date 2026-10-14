#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/optimisation/batch/small_batch6.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/optimisation/batch/small_accum.toml "$@"
