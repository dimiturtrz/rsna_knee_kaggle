# A no-op stand-in for lib.sh, so an arm can be READ without being run.
#
# Same function names, none of the effects: no GPU, no lock, no data, no training. `run_train` prints
# the flags it was handed instead of spending an hour proving them. That is what lets `kneemri.arm_check`
# hand every committed arm's invocation to the real argument parser before a queue is armed.
#
# It sources like the real thing because the arms `source "${ARM_LIB:-...}"` — the same seam
# `run_snapshot` already uses to point a snapshot copy back at the repo's lib.

set -u

REPO="$(pwd)"
PROCESSED="/tmp/arm_check"
LABELS="/tmp/arm_check/labels.csv"
LOGS="/tmp/arm_check/experiments"

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
