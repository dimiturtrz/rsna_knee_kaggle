#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/control/datasize/half_data.toml "$@"
