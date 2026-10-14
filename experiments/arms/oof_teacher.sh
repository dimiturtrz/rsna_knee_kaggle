#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/distillation/oof_teacher/oof_teacher.toml "$@"
