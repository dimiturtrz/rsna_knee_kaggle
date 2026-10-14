#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/grand_refit/grand_calib_span.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/composite/grand_refit/grand_calib_span_ff.toml "$@"
