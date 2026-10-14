#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/optimisation/lpft/band_lpft_head.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/optimisation/lpft/band_lpft.toml "$@"
