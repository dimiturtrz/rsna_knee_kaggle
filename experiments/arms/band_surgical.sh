#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/optimisation/lpft/band_early.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/optimisation/lpft/band_mid.toml "$@"
