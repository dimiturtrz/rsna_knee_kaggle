#!/usr/bin/env bash
# Waits for mricore_rate to finish, then runs extended_band — the SEE-axis arm, bead knee-q2yz7 (open P1).
#
# WHY THIS ARM NEXT. Its 89.6 GB cache (`images_336px_20sl_0.1-0.9band_extended.npy`) was built 06:39Z and
# has never been read, on a disk at 98%. Running it is what converts a stranded 89.6 GB into a reading;
# deleting it to reclaim the space would throw away the decode instead. `arm_check --only extended_band`
# reads 0 refused, 0 uncached. Its confound was already priced and is redundant: the three presence bits
# fitted alone read 0.5431 cross-fit, and on top of the member they are worth -0.0021, so a positive delta
# cannot be attributed to protocol metadata.
#
# THE MARKER IS SPELLED WITHOUT THE SEED. `_mark` (lib.sh:642) writes the run_train NAME while `_arm_log`
# (:685) names the FILE for the checkpoint basename, so `--out .../mricore_rate_s2026.pt` writes
# `mricore_rate_s2026.full.log` containing `ARM DONE mricore_rate`.
#
# THE ARM LOG IS UNDER $MODELS/experiments/, NOT logs/ — lib.sh re-runs an arm from a snapshot and the
# snapshot's tee writes there. A watcher gated on a path with no producer waits forever with the card idle
# behind it, and queue_check reads OK the whole time.
#
# The pid arm of the OR is the point: the marker alone hangs forever on a silent exit-0 death. `ps -W`
# because the MSYS pid is a different number from the Windows one and `kill -0` reads the wrong table.
set -u
cd "$(dirname "$0")/.."

MARKER="ARM DONE mricore_rate"
ARM_LOG="D:/models/rsna_knee_abnormality/experiments/mricore_rate_s2026.full.log"
PID="${1:-}"

while true; do
  grep -q "$MARKER" "$ARM_LOG" 2>/dev/null && { echo "marker seen $(date -u +%FT%TZ)"; break; }
  if [ -n "$PID" ] && ! ps -W | grep -q "[^0-9]$PID[^0-9]"; then
    echo "pid $PID gone without a marker $(date -u +%FT%TZ)"; break
  fi
  sleep 60
done

bash experiments/arms/extended_band.sh
echo "QUEUE extended_band exited $? $(date -u +%FT%TZ)"
