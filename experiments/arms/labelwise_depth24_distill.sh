#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/readout/labelwise_depth24/labelwise_depth24_distill.toml "$@"
