#!/usr/bin/env bash
# run.sh <experiments/recipes/<area>/<family>/<run>.toml> [flag ...] — train one recipe the way an arm does.
#
# The recipe IS the arm: its options, its `extends` trunk and its `[meta]` live in the TOML, so this
# script carries no recipe of its own. The run is named for the leaf file (`<run>`) and writes
# `$MODELS/<meta.out>.pt`. `out` is read from the leaf's own `[meta]` with awk rather than python,
# because arm_check runs this under a stub lib that shadows `uv`. Extra flags go to the trainer after the
# recipe, so they win over it.
recipe="${1:?usage: run.sh <experiments/recipes/<area>/<family>/<run>.toml> [flag ...]}"
shift
ARM_NAME="$(basename "$recipe" .toml)"
export ARM_NAME
source "${ARM_LIB:-$(dirname "$0")/lib.sh}"
out="$(awk '/^\[/{m=($0=="[meta]")} m && /^out *=/{gsub(/^out *= *"|".*$/,""); print}' "${ARM_SRC:-$REPO}/$recipe")"
require "$LABELS"
run_train "$ARM_NAME" --config "$recipe" "$@" --labels "$LABELS" --out "$MODELS/${out:-$ARM_NAME}.pt"
