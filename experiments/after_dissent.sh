#!/usr/bin/env bash
# Waits for foreign_dissent to exit, then re-runs the FINE band sweep that never ran, then mix30.
#
# WHY THE SWEEP IS HERE AND NOT BEHIND. `after_mix07_fine.sh` put the fine sweep AHEAD of dissent and
# it died in one line: `foreign_decode_fine.sh` inherits `require "$HOLDOUT"` from
# `foreign_decode_sweep.sh` but not the `HOLDOUT=` that sets it, so under `set -u` the arm dies at the
# require, before the first band, and the chain walked straight on to dissent with the card. Fixed in
# the arm (2026-09-06T19:27Z). The sweep is decode-only, ~15 min, and it is what the band choice and
# the equal-view TTA pre-registration both wait on.
#
# THEN mix30 — the supervision axis at its real endpoint, 3.0, because the distill term is additive
# (objective.py:411) and 1.0 is not one. ~73 min. Ordered behind the sweep because the sweep is on a
# clock (a clearing band's kernel has to be pushed for the 22:30Z census) and mix30 is not.
#
# The wait is on the WINDOWS pid card_check reported, tested with tasklist: an MSYS pid tool reads a
# Windows python as dead and would start a second job on a card a live arm still holds.
set -u
cd "$(dirname "$0")/.."

[ $# -ge 1 ] || { echo "usage: $(basename "$0") <windows-pid to wait on>" >&2; exit 2; }

until ! tasklist //FI "PID eq ${1}" 2>/dev/null | grep -q "${1}"; do sleep 60; done
echo "ARM DONE foreign_dissent_s2026 — starting the queue $(date -u +%FT%TZ)"

bash experiments/arms/foreign_decode_fine.sh
echo "QUEUE fine sweep exited $? $(date -u +%FT%TZ)"

bash experiments/arms/foreign_distill_mix30.sh
echo "QUEUE mix30 exited $? $(date -u +%FT%TZ)"
