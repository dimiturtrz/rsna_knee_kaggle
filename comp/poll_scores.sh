#!/usr/bin/env bash
# poll_scores.sh — watch the board until every pending entry resolves.
#
# A code submission takes HOURS to score (~8 h measured), which is longer than any single session sits
# still, and a pending entry blocks nothing — so the only cost of not watching is finding out late that
# a pre-registration was right or wrong. This existed twice as an inline loop typed at a prompt, and
# both times it died with the shell that launched it and left the last line reading PENDING, which is
# indistinguishable from "still waiting" until someone checks the process table.
#
# BOUNDED, and it says so when it gives up: a loop that exits silently at its cap reads exactly like a
# loop still running. `--polls` times `--every` is the window it covers; the default covers ~8 h.
#
# The CSV form rather than the table: the table truncates the description column, and the description
# is the only place an entry's identity is written down.
set -u

COMPETITION="${COMPETITION:-rsna-knee-abnormality-detection}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE="$REPO/.venv/Scripts/kaggle.exe"
EVERY=600
POLLS=48

while [ $# -gt 0 ]; do
  case "$1" in
    --every) EVERY="$2"; shift 2 ;;
    --polls) POLLS="$2"; shift 2 ;;
    *) echo "usage: poll_scores.sh [--every SECONDS] [--polls N]" >&2; exit 2 ;;
  esac
done

for poll in $(seq 1 "$POLLS"); do
  echo "-- poll $poll --"
  "$KAGGLE" competitions submissions -c "$COMPETITION" -v 2>/dev/null | tail -n +2 | head -6
  # THE EXIT CONDITION IS THE POINT: stop as soon as nothing is pending, so the log's last poll is the
  # one that carries the answer rather than the one that hit the cap.
  if ! "$KAGGLE" competitions submissions -c "$COMPETITION" -v 2>/dev/null | tail -n +2 | grep -q "pending"; then
    echo "ALL RESOLVED at poll $poll"
    exit 0
  fi
  sleep "$EVERY"
done
echo "POLL CAP REACHED at $POLLS polls — entries above are still pending, relaunch to keep watching"
