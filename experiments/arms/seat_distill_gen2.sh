#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/seat/seat_distill_gen2.toml "$@"
