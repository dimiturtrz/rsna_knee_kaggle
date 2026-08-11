#!/usr/bin/env bash
# after.sh <completion-marker> <file-to-watch> <arm.sh> [more-arms.sh...]
#
# Run arms, in order, once something else finishes. MEASURED 2026-08-11: one base arm with max slice
# pooling holds 21.6 GB of the 32 GB card, and two of them together drove it to 31.5 GB - so arms are
# serialised by default. Channel-stacked small arms are about half that and will pair up.
#
# The marker is a string the producer writes when it is DONE. Never gate on an output path existing -
# that fires when the producer starts.
#
#   nohup bash experiments/after.sh "ARM DONE compose_base" \
#     "$(...)/experiments/compose_base.full.log" experiments/arms/whole_stack.sh &
#
# The marker may instead be `pid:<n>`, which gates on that process exiting and ignores <file-to-watch>.
# Markers are written by an arm's SHELL, so a trainer that outlives its shell never announces itself -
# gating on the PID is the only thing left that is true. Second choice, not first: a PID says a
# process ended, never that it succeeded.
#
# An arm may carry its own arguments as ONE argv element: "experiments/arms/resolution.sh 448".
source "$(dirname "$0")/lib.sh"

MARKER="$1"; WATCH="$2"; shift 2
for entry in "$@"; do
  require "${entry%% *}"   # the arm path is the entry up to its first argument
done

echo "waiting for '$MARKER' in $WATCH before running: $*"
case "$MARKER" in
  pid:*) wait_for_pid "${MARKER#pid:}" ;;
  *) wait_for "$MARKER" "$WATCH" ;;
esac
for entry in "$@"; do
  echo "=== $entry ==="
  # Unquoted on purpose: an entry is "<arm.sh> [args...]" and has to word-split into run_snapshot,
  # which forwards everything after the arm path to the arm itself.
  # shellcheck disable=SC2086
  run_snapshot $entry
done
