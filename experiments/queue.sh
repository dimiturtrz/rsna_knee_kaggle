#!/usr/bin/env bash
# queue.sh <sentinel.pt> <entry> [entry...] — run arms serially once the GPU frees up.
#
# AN ENTRY IS AN ARM NAME PLUS THAT ARM'S ARGUMENTS, quoted as one word — `"resolution 420 448"`, the
# same shape after.sh takes. It used to be a bare name and the arguments were dropped in silence, so
# queueing a parameterised arm ran its DEFAULT: `resolution` spent a slot re-fitting a member we
# already had and overwrote its checkpoint with a copy of itself, while the queue file read as if it
# had asked for a rung (knee-8hu7). Splitting the entry is why `parts` exists rather than a loop over
# "$@" directly. An entry naming a `.toml` is a recipe and runs through `experiments/run.sh`. Snapshotting is not this script's job: lib.sh re-execs any arm launched from the repo
# tree out of a snapshot copy and forwards the arguments it was given.
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
#   bash experiments/queue.sh "$MODELS/weight_average_s2026.pt" depth_regions_narrow unfreeze_depth
#
# Each arm gets its own hard timeout so a wedged run cannot hold the queue: a 12-epoch arm is ~70 min
# at the current 360 s/epoch, and the ceiling is set well above that rather than near it, because a
# timeout that trips on a slow-but-healthy run destroys the arm and the slot both.
set -u

SENTINEL="${1:?usage: queue.sh <sentinel.pt> <entry> [entry...]}"
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

entries=("$@")
for entry in "${entries[@]}"; do
  read -r -a parts <<< "$entry"
  arm="${parts[0]}"
  case "$arm" in
    *.toml) script="experiments/run.sh"; named="$arm"; args=("${parts[@]}") ;;
    *) script="experiments/arms/$arm.sh"; named="$script"; args=("${parts[@]:1}") ;;
  esac
  [ -f "$named" ] || { echo "queue.sh: no such arm '$arm'"; exit 3; }
  echo "=== $(date -u +%H:%M:%SZ) queue: $entry ==="
  timeout "$ARM_TIMEOUT" bash "$script" "${args[@]}"
  echo "=== $(date -u +%H:%M:%SZ) queue: $arm exited $? ==="
done
