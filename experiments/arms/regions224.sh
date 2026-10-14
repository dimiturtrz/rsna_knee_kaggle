#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/readout/pooling/regions224.toml "$@"
