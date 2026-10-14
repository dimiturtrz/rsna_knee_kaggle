#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/encoder/dinov3_foreign/foreign_recipe_v3small_336_20sl.toml "$@"
