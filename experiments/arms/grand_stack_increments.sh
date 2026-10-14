#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/grand_increments/grand_stack_increments.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/grand_increments/grand_stack_increments_ff.toml "$@"
