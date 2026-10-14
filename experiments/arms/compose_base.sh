#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/compose/compose_base.toml "$@"
