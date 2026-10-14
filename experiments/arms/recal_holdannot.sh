#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/targets/recal/recal_holdannot.toml "$@"
