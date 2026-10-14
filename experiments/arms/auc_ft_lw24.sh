#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/targets/auc/auc_ft_lw24.toml "$@"
