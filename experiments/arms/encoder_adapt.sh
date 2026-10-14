#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/encoder/adapt_stage/encoder_adapt.toml "$@"
