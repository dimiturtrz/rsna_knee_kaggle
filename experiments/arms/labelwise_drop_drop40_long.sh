#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/readout/labelwise/labelwise_drop_drop40_long.toml "$@"
