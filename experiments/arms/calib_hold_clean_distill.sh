#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/targets/clean/calib_hold_clean_distill.toml "$@"
