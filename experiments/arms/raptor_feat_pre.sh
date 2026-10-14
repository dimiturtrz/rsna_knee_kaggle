#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/distillation/raptor/raptor_feat_pre.toml "$@"
