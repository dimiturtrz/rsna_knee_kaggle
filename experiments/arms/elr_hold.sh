#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/targets/elr/elr_hold_w0.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/targets/elr/elr_hold_w3.toml "$@"
