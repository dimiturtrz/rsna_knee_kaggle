#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/readout/depth/labelwise_depth20_b10.toml "$@"
