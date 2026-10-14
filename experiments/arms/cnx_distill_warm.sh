#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/encoder/convnext_warm/cnx_distill_warm.toml "$@"
