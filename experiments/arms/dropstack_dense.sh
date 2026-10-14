#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/readout/dropstack/dropstack_dense.toml "$@"
