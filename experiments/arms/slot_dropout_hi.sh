#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/readout/slot_dropout/slot_dropout_hi.toml "$@"
