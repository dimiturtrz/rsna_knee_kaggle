#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/readout/labelwise/labelwise_ext_drop40_long_ema.toml "$@"
