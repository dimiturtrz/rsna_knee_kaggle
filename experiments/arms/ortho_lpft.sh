#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/optimisation/lpft/ortho_lpft_head.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/optimisation/lpft/ortho_lpft.toml "$@"
