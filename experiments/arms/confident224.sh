#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/targets/blend_labels/confident224.toml "$@"
