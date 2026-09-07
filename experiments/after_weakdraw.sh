#!/usr/bin/env bash
# Wait for hisrecipe_distill to release the card, then run the row-6 Fracture draw pair.
#
# WHY NEITHER A MARKER NOR A pid WAIT ALONE. `hisrecipe_distill` never calls `run_train` — it is a
# `foreign_run` arm — so `_mark` never fires and no `ARM DONE hisrecipe_distill` line will ever be
# written to any file. A watcher grepping for one waits forever with the card idle behind it, and
# `queue_check` reads OK the whole time. The trainer's OWN success marker is the OOF artefact it
# writes at the end (`raptor_ft_<tag>_swa.holdout.npz`, seen for `repl2x2_c`), so that is the primary
# gate. But an artefact-only gate has the same idle failure in the other direction: a trainer that
# dies writes nothing, and `foreign_run.py:71` does not translate a death into a non-zero status, so
# a silent exit-0 death would hold this watcher forever. So the wait ends on EITHER the artefact
# appearing or the launching process leaving the process table, and the run below happens regardless
# — the card is the thing being protected, not the read.
#
# THE pid IS THE WINDOWS pid, read from `card_check`, so liveness is tested with `ps -W` and never
# with `kill -0` (an MSYS pid is a different number, and testing the wrong one reads a live process
# as dead).
#
# WHAT RUNS: the row-6 pair, ruled by mastermind and needing no read to start, so no branch decision
# taken while nobody is awake can block it. The D81 branch arm is armed separately, behind this one.
set -u
cd "$(dirname "$0")/.."

PID="${1:?usage: after_weakdraw.sh <windows pid of hisrecipe_distill>}"
OOF="D:/models/rsna_knee_abnormality/raptor_ft_hisrecipe_distill_swa.holdout.npz"

echo "waiting on hisrecipe_distill pid $PID / $OOF at $(date -u +%FT%TZ)"
until [ -f "$OOF" ] || ! ps -W | grep -q "[^0-9]$PID[^0-9]"; do sleep 60; done
echo "card released at $(date -u +%FT%TZ); artefact present: $([ -f "$OOF" ] && echo yes || echo no)"

bash experiments/arms/weakdraw.sh
echo "QUEUE weakdraw exited $? $(date -u +%FT%TZ)"
