#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/optimisation/augmentation/mixup224_long.toml "$@"
