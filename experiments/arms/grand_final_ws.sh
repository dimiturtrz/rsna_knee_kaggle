#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/grand_refit/grand_final_ws.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/grand_refit/grand_final_ws_ff.toml "$@"
