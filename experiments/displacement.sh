#!/usr/bin/env bash
# displacement.sh <member_stem> — the bar that replaced "peak8 + it > 0.8731".
#
# Appending a ninth member to the greedy peak has a 0/33 pass rate, so it prices nothing. This drops
# the candidate into the FULL same-split pool and re-runs greedy from scratch: keep it only if the new
# peak beats 0.8731, whether or not the set it comes back with looks like the old one.
set -euo pipefail

source "$(dirname "$0")/roots.sh"
D="$PROCESSED"
CANDIDATE="${1:?usage: displacement.sh <member_stem>}"

test -f "$D/${CANDIDATE}.holdout.npz" || { echo "no holdout record for $CANDIDATE"; exit 1; }

.venv/Scripts/python.exe -m kneemri.evaluation.record_mix --seats greedy \
  --records "$D"/*_s2026.pt "$D/${CANDIDATE}.pt" 2>&1 | tail -25
