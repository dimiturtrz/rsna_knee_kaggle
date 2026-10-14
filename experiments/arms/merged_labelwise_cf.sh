#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/merged/merged_labelwise_cf.toml "$@"
