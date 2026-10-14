#!/usr/bin/env bash
# run.sh <experiments/recipes/<area>/<family>/<run>.toml> [flag ...] — train one recipe the way an arm does.
#
# The recipe IS the arm: its options, its `extends` trunk and its `[meta]` live in the TOML, so this
# script carries no recipe of its own. The run is named for the leaf file (`<run>`) and writes
# `$MODELS/<meta.out>.pt`. The leaf's own `out`, `requires` and `[run] labels` are read with awk rather
# than python, because arm_check runs this under a stub lib that shadows `uv`; their `$MODELS` and
# `$PROCESSED` are the roots lib.sh sets. Every input the leaf requires must exist before the card is
# spent. Extra flags go to the trainer after the recipe, so they win over it. Each `[meta] after` entry is
# a `kneemri` module and its arguments, run once the checkpoint is written (a soup, a backfill, the
# arm's own reading); they are word-split, never evaluated, so a leaf can name a step but not a script.
recipe="${1:?usage: run.sh <experiments/recipes/<area>/<family>/<run>.toml> [flag ...]}"
shift
ARM_NAME="$(basename "$recipe" .toml)"
export ARM_NAME
source "${ARM_LIB:-$(dirname "$0")/lib.sh}"

leaf() {
  # Every quoted string `key` holds in the leaf's own `[table]`, one per line, roots filled.
  awk -v table="[$1]" -v key="$2" '/^\[/{m=($0==table)} m && $1==key {n=split($0,p,"\""); for(i=2;i<n;i+=2) print p[i]}' \
    "${ARM_SRC:-$REPO}/$recipe" | sed -e "s|\$MODELS|$MODELS|g" -e "s|\$PROCESSED|$PROCESSED|g"
}

out="$(leaf meta out)"
labels="$(leaf run labels)"
labels="${labels:-$LABELS}"
require "$labels"
while IFS= read -r input; do require "$input"; done < <(leaf meta requires)
run_train "$ARM_NAME" --config "$recipe" "$@" --labels "$labels" --out "$MODELS/${out:-$ARM_NAME}.pt" || exit
while IFS= read -r step; do
  # shellcheck disable=SC2086
  uv run --no-sync --project "$REPO" python -m $step || exit
done < <(leaf meta after)
