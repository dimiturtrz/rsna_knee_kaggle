#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/distillation/oof/oof_fold0.toml "$@"
