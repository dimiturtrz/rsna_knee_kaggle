#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/optimisation/learning_rate/headlr_floor.toml "$@"
