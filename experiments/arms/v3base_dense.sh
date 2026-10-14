#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/encoder/dinov3_base/v3base_dense.toml "$@"
