#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/readout/depth/labelwise_depth24_drop40_long.toml "$@"
