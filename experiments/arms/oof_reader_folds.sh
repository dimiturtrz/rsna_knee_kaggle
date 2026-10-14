#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/oof/oof_reader_fold0.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/oof/oof_reader_fold1.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/oof/oof_reader_fold2.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/oof/oof_reader_fold3.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/oof/oof_reader_fold4.toml "$@"
