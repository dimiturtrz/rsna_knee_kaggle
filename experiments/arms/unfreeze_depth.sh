#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/encoder/depth/unfreeze_4.toml "$@"
