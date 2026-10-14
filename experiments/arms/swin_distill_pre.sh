#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/encoder/feature_pre/swin_distill_pre.toml "$@"
