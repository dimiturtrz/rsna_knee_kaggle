#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/oof/oof_band_fold0.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/oof/oof_band_fold1.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/oof/oof_band_fold2.toml "$@" \
  && bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/oof/oof_band_fold3.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/oof/oof_band_fold4.toml "$@"
