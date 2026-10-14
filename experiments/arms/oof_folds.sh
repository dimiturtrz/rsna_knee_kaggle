#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/oof/oof_fold1.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/oof/oof_fold2.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/oof/oof_fold3.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/oof/oof_fold4.toml "$@"
