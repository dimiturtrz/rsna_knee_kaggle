#!/usr/bin/env bash
# Wait for `ourtrunk44` to release the card, then run the `--corpus-oversample` instrument.
#
# WHY THE GATE IS A PID AND NOT A MARKER. `ourtrunk44` is a `foreign_run` arm. `_mark` lives in
# `run_train`, so a `foreign_run` arm writes NO `ARM DONE` line anywhere, ever — a watcher grepping for
# one waits forever with the card idle behind it, and `queue_check` reads OK the whole time because a
# live process gating an owed arm is all it can see. The gate is Windows-pid liveness via `ps -W`,
# never `kill -0` (the MSYS pid is a different number).
#
# WHICH PID. 10684, the 10:48:33Z launch — the oldest of THREE identical `ourtrunk44` processes that
# were sharing this card and the same `--tag`, i.e. clobbering each other's artefacts as well as
# each other's throughput. The younger two were killed at 08:12Z.
#
# WHY THIS ARM AND NOT `merged_chain_span`. `merged3_oversample` is 40 min against chain_span's 3 h,
# it sets a flag that 0 of 292 arm scripts have ever set (D189 §6, word-boundary census), its control
# is already on disk (`merged3_corpus_s2026` = 0.8521), and it decides whether per-corpus head
# conditioning gets scoped as a build at all. chain_span's own pre-registration puts P(clears) < 0.05.
# chain_span's waiter was killed at 08:12Z so it could not take the card first.
set -u

cd /d/personal_projects/rsna_knee_kaggle || exit 1

PID=10684
LOG="logs/after_ourtrunk44.log"

echo "waiting for windows pid $PID to exit ($(date -u +%Y-%m-%dT%H:%M:%SZ))" >> "$LOG"
while ps -W | grep -q "[^0-9]$PID[^0-9]"; do sleep 30; done
echo "card released at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

bash experiments/arms/merged3_oversample.sh >> "$LOG" 2>&1
echo "merged3_oversample finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
