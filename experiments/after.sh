#!/usr/bin/env bash
# after.sh [--smoke <dir>] <completion-marker> <file-to-watch> <arm.sh> [more-arms.sh...]
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
# CHECK THAT ENTRY BEFORE QUEUEING IT, in the same spelling — arm_check runs the arm files bare, so an
# argument-carrying rung is otherwise queued on flags nothing has parsed or smoked:
#
#   python -m kneemri.experiments.arm_check --as-queued "experiments/arms/resolution.sh 448 448 4e-5 9" --smoke <dir>
#
# That check earned itself on its first run: the entry spelled "resolution.sh 448" - this comment's own
# example until 2026-08-17 - OOMed in seconds, because the arm kept batch 12 at a size knee-d2fc had
# measured as needing 9.
#
# --smoke <dir> RUNS EACH ARM'S SMOKE FIRST, on the card the producer just freed, and abandons the queue
# if it fails. A recipe is supposed to be smoked before it is queued for forty minutes, but the card is
# usually BUSY at the moment of queueing - which is the whole reason there is something to queue behind -
# so the smoke has to happen after the wait, not before it. Without this the only way to gate a run on
# its own smoke was a hand-rolled `nohup bash -c 'wait_for ...; arm_check --smoke ... && arm.sh'`, and
# such a chain is invisible to `queue_check`, which enumerates watchers by globbing logs/after_*.log for
# the "waiting for" line below - so it reported every owed arm gated while owing one it could not see
# (knee-61ez).
source "$(dirname "$0")/lib.sh"

SMOKE=""
if [ "$1" = "--smoke" ]; then SMOKE="$2"; shift 2; fi

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
  # --as-queued takes the entry in the SPELLING it will run in, arguments included, which is the only
  # spelling worth smoking. The directory is per-arm because arm_check refuses one that already exists.
  [ -n "$SMOKE" ] && { uv run --no-sync python -m kneemri.experiments.arm_check \
    --as-queued "$entry" --smoke "$SMOKE/$(basename "${entry%% *}" .sh)" --smoke-studies 60 || exit 1; }
  # Unquoted on purpose: an entry is "<arm.sh> [args...]" and has to word-split into run_snapshot,
  # which forwards everything after the arm path to the arm itself.
  # shellcheck disable=SC2086
  run_snapshot $entry
done
