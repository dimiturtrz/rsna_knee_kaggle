#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/encoder/mrnet/mrnet_warm.toml "$@"
