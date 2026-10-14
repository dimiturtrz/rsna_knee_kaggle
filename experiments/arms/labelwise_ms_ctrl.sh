#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/readout/labelwise/labelwise_ms_ctrl.toml "$@"
