#!/usr/bin/env bash
# Waits out fovjitter160_reflect.sh, then puts the CARVE adaptation readout on the card in its place.
#
# The gate is the ARM'S OWN INTERPRETER, not a marker line, for two reasons that have each cost a wake.
# `_mark` writes `ARM DONE <run_train name>` into a file named for the CHECKPOINT, and the two spellings
# are different variables — so a watcher grepping the checkpoint spelling in the marker waits forever
# with the card idle behind it, and `queue_check` reports OK the whole time. And the log is under
# `$MODELS/experiments/`, not `logs/`, which is the other half of the same trap.
#
# `.venv-arm` is used by no other launcher on this machine, so its absence from the process table means
# the card is free under every ending — a clean finish, a crash, or a kill. `ps -W` is required: MSYS
# pids are a different number space and `kill -0` reads a live arm as dead.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 1

while ps -W 2>/dev/null | grep -qE 'venv-arm[\\/]Scripts[\\/]python'; do
  sleep 30
done

echo "the card is free at $(date -u +%Y-%m-%dT%H:%M:%SZ); arming carve_adapt_read"

UV_NO_SYNC=1 uv run python -m kneemri.experiments.card_check || {
  echo "card_check refused — not arming"
  exit 1
}

# THE CONTROL FIRST, ON THE SAME CARD IN THE SAME HOUR. `fovjitter160_read.sh` is `carve_adapt_read.sh`
# minus the single flag `--adapt`, over the same three checkpoints and the same foreign corpus, and it
# has never been run — so there is no remembered figure to compare against and the pair has to be taken
# together. Sequential rather than backgrounded: two readouts on one card would contend, and contention
# is what turned a three-minute member into eleven once already.
bash experiments/arms/fovjitter160_read.sh > logs/fovjitter160_read.log 2>&1
echo "control finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) with status $?"

bash experiments/arms/carve_adapt_read.sh > logs/carve_adapt_read.log 2>&1
echo "carve_adapt_read finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) with status $?"
