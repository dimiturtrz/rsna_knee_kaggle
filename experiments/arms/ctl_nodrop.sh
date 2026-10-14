#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/ctl/nodrop.toml "$@"
