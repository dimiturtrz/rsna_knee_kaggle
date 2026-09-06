#!/usr/bin/env bash
# submit_at_reset.sh <owner/kernel> <version> "<message>" — retry a code submission until a slot exists.
#
# Slots reset at 00:00 UTC and are destroyed, never banked. A kernel that is pushed, COMPLETE and
# census-verified while the day's five are spent is not a skip: it is an entry waiting on a clock. The
# CLI answers "0 submissions remaining today" and exits, so the retry IS the wait.
set -u
SLUG="${1:?usage: submit_at_reset.sh <owner/kernel> <version> <message>}"
VERSION="${2:?version}"
MESSAGE="${3:?message}"
COMP=rsna-knee-abnormality-detection
cd "$(dirname "${BASH_SOURCE[0]}")/.."
until uv run --no-sync --with kaggle kaggle competitions submit "$COMP" -k "$SLUG" -v "$VERSION" -f submission.csv -m "$MESSAGE" 2>&1 | tee /dev/stderr | grep -q "Successfully submitted"; do
  sleep 120
done
