#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/encoder/dino/large_labelwise224_dense.toml "$@"
