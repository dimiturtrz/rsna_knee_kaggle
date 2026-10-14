#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/teacher/oof_gen2_fold0.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/teacher/oof_gen2_fold1.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/teacher/oof_gen2_fold2.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/teacher/oof_gen2_fold3.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/teacher/oof_gen2_fold4.toml "$@"
