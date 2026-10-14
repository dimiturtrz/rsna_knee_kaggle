#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/encoder/convnext/coatnet_hybrid.toml "$@"
