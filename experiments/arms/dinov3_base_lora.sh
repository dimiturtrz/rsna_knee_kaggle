#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/encoder/dinov3_base/dinov3_base_lora.toml "$@"
