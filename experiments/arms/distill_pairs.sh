#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/distillation/teacher/distill_pairs.toml "$@"
