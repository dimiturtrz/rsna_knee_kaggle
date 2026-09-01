#!/usr/bin/env bash
# watch.sh — emit one line per tick naming the letters that arrived in inbox/, then file them.
#
# Armed once per session under Claude Code's Monitor tool, which turns each stdout line into a
# notification. Nothing else consumes this: a harness without Monitor should read the directory
# directly rather than run a watcher whose output goes nowhere.
#
#   tool:        Monitor
#   command:     bash inbox/watch.sh
#   persistent:  true
#
# THE ECHO COMES BEFORE THE MOVE, deliberately. Filing first and echoing after loses any letter that
# lands between the listing and the move — the tick reports what it saw, then files exactly that.
#
# Bodies are not echoed. A letter can be any length and the notification is a line, so the tick names
# the file and the reader cats it out of inbox/read/.
set -u

INBOX="$(cd "$(dirname "$0")" && pwd)"
READ="$INBOX/read"
POLL_SECONDS=15

mkdir -p "$READ"

while true; do
  arrived=$(cd "$INBOX" && ls *.txt 2>/dev/null | sort | tr '\n' ' ')
  if [ -n "$arrived" ]; then
    echo "inbox: $arrived (cat inbox/read/<file> for bodies)"
    for letter in $arrived; do
      mv -f "$INBOX/$letter" "$READ/$letter" 2>/dev/null || true
    done
  fi
  # Backgrounded so a SIGTERM lands during the interval rather than after it: a foreground `sleep`
  # blocks bash's signal handling until it returns, which makes the watcher take up to a full poll
  # to die and orphans it past the session that armed it.
  sleep "$POLL_SECONDS" &
  wait $! 2>/dev/null || true
done
