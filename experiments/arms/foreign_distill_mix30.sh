#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/distillation/foreign_mix/foreign_distill_mix30.toml "$@"
