#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/targets/foreign_labels/warm_hislabels.toml "$@"
