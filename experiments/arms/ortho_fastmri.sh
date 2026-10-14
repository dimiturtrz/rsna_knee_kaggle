#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/encoder/fastmri/ortho_fastmri.toml "$@"
