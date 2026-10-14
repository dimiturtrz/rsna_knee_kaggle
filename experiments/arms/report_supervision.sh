#!/usr/bin/env bash
bash "$(dirname "$0")/../run.sh" experiments/recipes/targets/report_text/report_0.3.toml "$@" \
  && exec bash "$(dirname "$0")/../run.sh" experiments/recipes/targets/report_text/report_1.0.toml "$@"
