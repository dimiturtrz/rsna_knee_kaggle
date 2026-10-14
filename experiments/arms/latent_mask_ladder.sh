#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/encoder/latent/latent_mask_dense.toml "$@"
