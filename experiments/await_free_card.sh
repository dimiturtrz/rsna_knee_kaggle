# Run a command only once NO FOREIGN TRAINER holds the card, however long that takes.
#
# lib.sh's own free-VRAM gate waits FOREIGN_WAITS minutes and then starts ANYWAY, loudly, accepting
# PCIe paging. That is the right default for a card that is ours: a queued arm that never runs is
# worse than a slow one. It is the wrong default when the co-tenant is the other project, which shares
# this card by arrangement rather than by our leave — barging pages THEIR run out, and the arrangement
# is worth more than any one arm of ours.
#
# So this waits without a ceiling and never starts anyway. The `--deadline` is a give-up, not a barge:
# it exits non-zero having run nothing, which is the outcome to prefer when the alternative is taking
# a card off a peer.
#
# It asks the DEVICE for compute processes rather than reading free VRAM, for the reason card_check
# documents: this machine keeps ~2.6 GB in window-manager and browser clients that appear in
# `memory.free` and never release it, so a free-VRAM test cannot pass on an idle card for a 20-slice
# recipe. A co-tenant is a PROCESS, and the driver names them.
set -u
source "${ARM_LIB:-$(dirname "$0")/lib.sh}"
deadline_minutes=600
interval=120
while [ $# -gt 0 ]; do
  case "$1" in
    --deadline) deadline_minutes="$2"; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
[ $# -gt 0 ] || { echo "usage: await_free_card.sh [--deadline MINUTES] -- <command...>" >&2; exit 2; }

ours="$(pwd)"
waited=0
while :; do
  # `C` exactly, never as a prefix: `C+G` is Explorer and Firefox, and `--query-compute-apps` on this
  # display driver enumerates every client of the device — it answered 23 on an idle-but-for-one-trainer
  # card the first time this script ran. The Type column exists in no CSV query, so `compute_tenants`
  # parses the human table for it. One home for that parse, in lib.sh, beside the gate that needs it.
  foreign="$(compute_tenants | wc -l)"
  if [ "$foreign" -eq 0 ]; then
    echo "card free after ${waited}m — starting: $*"
    cd "$ours" && exec "$@"
  fi
  if [ $((waited)) -ge "$deadline_minutes" ]; then
    echo "GIVING UP after ${waited}m: a foreign trainer still holds the card, and this arm is not worth barging for" >&2
    exit 1
  fi
  [ $((waited % 30)) -eq 0 ] && echo "${waited}m: ${foreign} compute process(es) on the card, still waiting"
  sleep "$interval"
  waited=$((waited + interval / 60))
done
