# Shared plumbing for the training arms. Source it; do not run it.
#
# An arm is a hypothesis, a control number, and one `kneemri.train` invocation. Everything an arm
# needs that is not the hypothesis itself lives here, so an arm file is only ever the interesting part.
#
# Two rules are encoded rather than remembered:
#   - The data root is resolved through `core.paths`, never hardcoded. paths.yaml is gitignored and
#     machine-specific; an arm that spells out D:/data is not runnable on another machine.
#   - A readiness gate keys on a marker the producer writes when it FINISHES. An output path proves a
#     job started and nothing more - `wait_for` exists so that mistake cannot be made per-arm again.

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PROCESSED="$(uv run python -c 'from pathlib import Path
from core.paths import DataRoot
print(DataRoot.from_config(Path("paths.yaml")).processed().as_posix())')"
LABELS="$PROCESSED/labels/report_labels_v2.csv"
LOGS="$PROCESSED/experiments"
mkdir -p "$LOGS"

require() {
  # Fail before spending GPU, not after. Every path an arm depends on goes through here.
  for path in "$@"; do
    [ -e "$path" ] || { echo "MISSING: $path"; exit 1; }
  done
}

run_train() {
  # $1 is the arm's name; the rest are kneemri.train flags.
  #
  # The FULL output is kept out of the repo under processed/experiments, and only the epoch and best
  # lines reach the terminal. Earlier arms piped straight into grep, which discarded any traceback
  # whose wording missed the filter - the tee is what makes a silent failure recoverable.
  local name="$1"; shift
  uv run python -u -m kneemri.train "$@" 2>&1 \
    | tee "$LOGS/$name.full.log" \
    | grep --line-buffered -E "epoch |best |rror|emory"
}

wait_for() {
  # wait_for <completion-marker> <file>. Blocks until the marker appears.
  local marker="$1" file="$2"
  until grep -q "$marker" "$file" 2>/dev/null; do sleep 60; done
}
