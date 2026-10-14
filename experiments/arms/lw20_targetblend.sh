#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/targets/teacher_blend/labelwise_dense_drop40_long_targetblend.toml "$@"
