#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/control/base/base_labelwise224_dense_drop40_long36.toml "$@"
