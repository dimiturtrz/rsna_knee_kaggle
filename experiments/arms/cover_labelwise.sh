#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/geometry/depth/cover_labelwise.toml "$@"
