#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/encoder/adaptation/unfreeze_deep.toml "$@"
