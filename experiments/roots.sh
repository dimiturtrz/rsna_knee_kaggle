# The data roots, resolved through `core.paths` from the gitignored paths.yaml. Source it; do not run it.
# Both roots out of ONE interpreter start: a second `uv run` costs a fresh import before a study is read.
_roots="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && uv run --no-sync python -c 'from pathlib import Path
from core.paths import DataRoot
root = DataRoot.from_config(Path("paths.yaml"))
print(root.processed().as_posix())
print(root.models().as_posix())
print(root.raw("skmtea").as_posix())')"
PROCESSED="$(echo "$_roots" | sed -n 1p)"
MODELS="$(echo "$_roots" | sed -n 2p)"
SKMTEA_RAW="$(echo "$_roots" | sed -n 3p)"
export PROCESSED MODELS
