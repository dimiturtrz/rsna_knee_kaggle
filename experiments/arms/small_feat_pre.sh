#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/encoder/feature_pre/small_feat_pre.toml "$@"
