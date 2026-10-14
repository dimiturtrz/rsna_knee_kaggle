#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/optimisation/learning_rate/decay224_high.toml "$@"
