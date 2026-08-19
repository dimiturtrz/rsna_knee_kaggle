#!/usr/bin/env bash
# queue.sh <sentinel.pt> <arm> [arm...] — run arms serially once the GPU frees up.
#
# WHY THIS EXISTS. Arms were launched one at a time, so the GPU sat idle for the whole gap between a
# run finishing and someone noticing (four hours on 2026-08-19). The fix is not a longer chain typed
# into a shell — that only moves the gap to the end of the chain — it is a queue that can be EXTENDED
# while the machine is busy. This script waits for the arm that is currently running, then takes over.
#
# THE SENTINEL IS THE RUNNING ARM'S OUTPUT CHECKPOINT, not a process id. A pid is wrong twice here:
# nvidia-smi under WDDM reports every process on the display adapter with `[N/A]` memory, so it cannot
# say which one is training, and a pid does not survive the shell that started it. A checkpoint appears
# exactly once, when the run it belongs to has written its last epoch.
#
#   bash experiments/queue.sh "$PROCESSED/weight_average_s2026.pt" depth_regions_narrow unfreeze_depth
#
# Each arm gets its own hard timeout so a wedged run cannot hold the queue: a 12-epoch arm is ~70 min
# at the current 360 s/epoch, and the ceiling is set well above that rather than near it, because a
# timeout that trips on a slow-but-healthy run destroys the arm and the slot both.
set -u

SENTINEL="${1:?usage: queue.sh <sentinel.pt> <arm> [arm...]}"
shift
[ "$#" -gt 0 ] || { echo "queue.sh: no arms named"; exit 2; }

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ARM_TIMEOUT="${ARM_TIMEOUT:-20000}"
POLL_SECONDS=60

while [ ! -f "$SENTINEL" ]; do
  sleep "$POLL_SECONDS"
done
# The checkpoint appears before the process has finished writing it and released the GPU.
sleep 120

for arm in "$@"; do
  script="experiments/arms/$arm.sh"
  [ -f "$script" ] || { echo "queue.sh: no such arm '$arm'"; exit 3; }
  echo "=== $(date -u +%H:%M:%SZ) queue: $arm ==="
  timeout "$ARM_TIMEOUT" bash "$script"
  echo "=== $(date -u +%H:%M:%SZ) queue: $arm exited $? ==="
done
