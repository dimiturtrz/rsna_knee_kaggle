#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/teacher/oof_clean_fold1.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/teacher/oof_clean_fold2.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/teacher/oof_clean_fold3.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/teacher/oof_clean_fold4.toml "$@"
