#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/encoder/convnext/cnx_mid_rate.toml "$@"
