#!/usr/bin/env bash
exec bash "$(dirname "$0")/../run.sh" experiments/recipes/encoder/mricore_corpus/mricore_corpus.toml "$@"
