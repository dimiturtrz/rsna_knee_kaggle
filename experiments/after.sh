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
source "$(dirname "$0")/lib.sh"

MARKER="$1"; WATCH="$2"; shift 2
require "$@"

echo "waiting for '$MARKER' in $WATCH before running: $*"
wait_for "$MARKER" "$WATCH"
for arm in "$@"; do
  echo "=== $arm ==="
  run_snapshot "$arm"
done
