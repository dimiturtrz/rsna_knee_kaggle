#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/encoder/dinov3_base/warm_v3base_lora.toml "$@"
