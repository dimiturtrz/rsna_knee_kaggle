#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/distillation/foreign/foreign_distill.toml "$@"
