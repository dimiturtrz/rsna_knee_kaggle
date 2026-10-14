#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/control/base/base_dense.toml "$@"
