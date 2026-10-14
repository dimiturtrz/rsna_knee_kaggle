#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/targets/label_versions/labelcut_v3_ema.toml "$@"
