# A no-op stand-in for lib.sh, so an arm can be READ without being run.
#
# Same function names, none of the effects: no GPU, no lock, no data, no training. `run_train` prints
# the flags it was handed instead of spending an hour proving them. That is what lets `kneemri.experiments.arm_check`
# hand every committed arm's invocation to the real argument parser before a queue is armed.
#
# It sources like the real thing because the arms `source "${ARM_LIB:-...}"` — the same seam
# `run_snapshot` already uses to point a snapshot copy back at the repo's lib.

set -u

# The shared recipe, so an arm and the checker see the same flags.
source "$(dirname "${BASH_SOURCE[0]}")/recipe.sh"

REPO="$(pwd)"
# The REAL processed root when the checker could resolve one, a fabricated path when it could not.
# A fabricated path is enough to prove the flags parse, and it is what CI gets — but `arm_check --smoke`
# actually RUNS the invocation, and a labels file at /tmp is a FileNotFoundError in every arm at once.
# The derivation below is lib.sh's, character for character, so the smoke reads the file the arm would.
PROCESSED="${ARM_CHECK_PROCESSED:-/tmp/arm_check}"
LABELS="$PROCESSED/labels/report_labels_v2.csv"
LOGS="$PROCESSED/experiments"

require() { :; }
hold_card() { :; }
release_card() { :; }
wait_for() { :; }
wait_for_pid() { :; }
wait_for_cache() { :; }
run_snapshot() { :; }

# A no-op `uv` AHEAD OF THE REAL ONE, because not every arm goes through run_train: the diversity
# curve and the cache build drive `uv run` directly, and under a stub lib they would do the real work.
# Worse, killing the arm on timeout leaves that grandchild holding the pipe, so the check hangs rather
# than skipping the file. Shadowing uv makes those arms no-ops instead.
_ARM_CHECK_BIN="$(mktemp -d)"
printf '#!/usr/bin/env bash
exit 0
' > "$_ARM_CHECK_BIN/uv"
chmod +x "$_ARM_CHECK_BIN/uv"
PATH="$_ARM_CHECK_BIN:$PATH"
export PATH

run_train() {
  # The arm's name, a tab, then its flags — one line, so the caller can split it without a parser.
  local name="$1"
  shift
  printf '%s\t%s\n' "$name" "$*"
}
