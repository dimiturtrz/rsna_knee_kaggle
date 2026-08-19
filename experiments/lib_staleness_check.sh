#!/usr/bin/env bash
# Does a shell know when it is holding an out-of-date lib.sh? `bash experiments/lib_staleness_check.sh`.
#
# The failure this guards is knee-yn6j: a chain sources lib.sh once and then runs arm after arm, so an
# arm launched after a mechanism landed still ran the lib.sh parsed at the chain's launch. Two arms
# finished with no SEAT VALUE line on 2026-08-19 and the mechanism looked broken when it was absent.
#
# It checks the DECISION, not the re-source: `run_train`'s refresh path holds the card and spends an
# arm, so proving it end to end costs a GPU slot. What is proved here is that a shell holding the
# current file reads fresh, and a shell holding any other content reads stale — which is the whole of
# what the refresh branches on.
set -u

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

failures=0

if _lib_is_stale; then
  echo "FAIL: a shell that just sourced the working tree's lib.sh reads as stale ($LIB_PARSED)"
  failures=$((failures + 1))
fi

LIB_PARSED="0000000000000000000000000000000000000000"
if ! _lib_is_stale; then
  echo "FAIL: a shell holding a different lib.sh reads as fresh — a chain would never refresh"
  failures=$((failures + 1))
fi

LIB_PARSED="$(_lib_blob)"
if _lib_is_stale; then
  echo "FAIL: re-reading the blob does not clear the staleness, so the refresh would not converge"
  failures=$((failures + 1))
fi

if [ "$failures" -ne 0 ]; then
  echo "lib staleness: $failures failure(s)"
  exit 1
fi
echo "lib staleness: fresh reads fresh, other content reads stale, re-reading converges"
