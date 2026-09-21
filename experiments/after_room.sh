#!/usr/bin/env bash
# Hold an arm until the CARD HAS ROOM FOR IT, then run it: after_room.sh <free-mib> <arm> [args...]
#
# NAMED `after_*` AND PRINTING `after.sh`'s HEADER ON PURPOSE. `queue_check` finds a watcher two ways
# and this script failed both: its process probe matches an `after*.sh` argv, and `Watch.of` refuses
# any log whose first line is not the `waiting for '<marker>' in <path> before running: <arms>` header.
# Under its own name, and printing its own prose, it was a live watcher owing an arm that the queue
# gate reported as `OK: nothing is queued` -- the precise blindness that gate exists to prevent. The
# marker here is a memory figure rather than a line in a file, so `<path>` is the instrument that
# answers for it.
#
# WHY THIS EXISTS. `card_check` answers CONTENTION -- how many of OUR jobs hold the card -- and says
# nothing about free memory. On 2026-09-21 it read `OK: at most one of our jobs holds this card` with
# peers at 19.6 GB of 32.6, and `transduce_dmlb` was launched into ~12.7 GB free behind a 22 GB cap.
# Three peer jobs then grew into the remainder, the card hit 31.6 GB at 99% util with five compute
# processes, and the arm was killed three minutes in. The check was not wrong; it was answering a
# different question than the one being asked of it. This script asks the missing one.
#
# WHY A CONSECUTIVE COUNT AND NOT A SINGLE READ. Peers on this card GROW -- they do not hold still --
# so one read showing room is a claim about an instant, not about the next hour. The gate requires the
# free figure to hold across STABLE_POLLS consecutive minutes, which is the cheapest available evidence
# that a peer is finished rather than between allocations.
#
# THE MARGIN IS THE CALLER'S. Pass the free MiB the arm actually needs INCLUDING headroom for peer
# growth, not the arm's `--vram-gb` cap: a cap is a ceiling the run may never reach, and the reserve is
# what it takes. Reading the reserve off a previous run of the same recipe is the honest source.
#
# CONTENTION IS STILL CHECKED, because room and contention are independent failures and the standing
# rule is that no second job of ours ever shares this card.
set -u
cd "$(dirname "$0")/.."

FREE_NEEDED="${1:?usage: after_room.sh <free-mib> <arm> [args...]}"
ARM="${2:?usage: after_room.sh <free-mib> <arm> [args...]}"
shift 2

STABLE_POLLS=3
POLL_SECONDS=60

free_mib() {
  local used total
  read -r used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr -d ',\r')
  echo $((total - used))
}

# THE MARKER IS THE BARE MiB FIGURE BECAUSE `queue_check` CROSSES IT AGAINST argv. A watcher is healthy
# only when a live process carries BOTH the marker and the arms its log names; a marker spelled in prose
# ('17000 MiB free x3') appears in no argv, so the check saw the log, saw the process, matched neither to
# the other, and printed STRANDED on a perfectly live queue -- which invites the relaunch that puts two
# watchers on one arm, the opposite hole that gate exists to close. The figure below is argv's own token.
echo "waiting for '${FREE_NEEDED}' in nvidia-smi free MiB x${STABLE_POLLS} before running: $ARM $*"
echo "held from $(date -u +%FT%TZ)"

stable=0
while [ "$stable" -lt "$STABLE_POLLS" ]; do
  sleep "$POLL_SECONDS"
  free=$(free_mib)
  if [ "$free" -ge "$FREE_NEEDED" ]; then
    stable=$((stable + 1))
  else
    stable=0
  fi
  echo "$(date -u +%FT%TZ) free ${free} MiB, stable ${stable}/${STABLE_POLLS}"
done

if ! UV_NO_SYNC=1 .venv/Scripts/python.exe -m kneemri.experiments.card_check; then
  echo "REFUSED $(date -u +%FT%TZ): card_check says one of ours already holds this card"
  exit 1
fi

echo "=== ROOM $(date -u +%FT%TZ) at $(free_mib) MiB free; launching $ARM"
bash "$ARM" "$@"
echo "QUEUE $(basename "$ARM") exited $? $(date -u +%FT%TZ)"
