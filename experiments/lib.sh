# Shared plumbing for the training arms. Source it; do not run it.
#
# An arm is a hypothesis, a control number, and one `kneemri.train` invocation. Everything an arm
# needs that is not the hypothesis itself lives here, so an arm file is only ever the interesting part.
#
# Two rules are encoded rather than remembered:
#   - The data root is resolved through `core.paths`, never hardcoded. paths.yaml is gitignored and
#     machine-specific; an arm that spells out D:/data is not runnable on another machine.
#   - A readiness gate keys on a marker the producer writes when it FINISHES. An output path proves a
#     job started and nothing more - `wait_for` exists so that mistake cannot be made per-arm again.

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PROCESSED="$(uv run python -c 'from pathlib import Path
from core.paths import DataRoot
print(DataRoot.from_config(Path("paths.yaml")).processed().as_posix())')"
LABELS="$PROCESSED/labels/report_labels_v2.csv"
LOGS="$PROCESSED/experiments"
mkdir -p "$LOGS"

run_snapshot() {
  # Run an arm from a COPY, never from the file in the repo.
  #
  # bash reads a script incrementally, by byte offset, as it executes. Editing that file mid-run makes
  # it resume at a stale offset - on 2026-08-11 a sweep that rewrote every arm's trailing marker
  # truncated whole_stack.sh under a running bash, which died with "unexpected EOF while looking for
  # matching quote" and took a 2-hour arm with it. The repo has to stay editable while long work runs,
  # so the run gets its own immutable copy.
  local arm="$1"; shift
  local snapshot="$LOGS/$(basename "$arm" .sh).running.sh"
  cp "$arm" "$snapshot"
  # "$@" is forwarded: an arm that takes arguments (which folds, which resolution) was being run with
  # none, so its loop iterated zero times and it exited 0 with an empty log. A silent no-op is the
  # worst failure available - the GPU simply sat idle and nothing reported anything.
  ARM_LIB="$REPO/experiments/lib.sh" bash "$snapshot" "$@"
}

require() {
  # Fail before spending GPU, not after. Every path an arm depends on goes through here.
  for path in "$@"; do
    [ -e "$path" ] || { echo "MISSING: $path"; exit 1; }
  done
}

run_train() {
  # $1 is the arm's name; the rest are kneemri.train flags.
  #
  # The FULL output is kept out of the repo under processed/experiments, and only the epoch and best
  # lines reach the terminal. Earlier arms piped straight into grep, which discarded any traceback
  # whose wording missed the filter - the tee is what makes a silent failure recoverable.
  #
  # THE STATUS IS THE TRAINER'S, NOT THE PIPELINE'S. A pipeline reports its LAST command, so without
  # PIPESTATUS a crashed trainer is reported by a contented `grep` and the arm goes on to announce it
  # finished. That is exactly what arm U did on 2026-08-11: it died at epoch 7 of 24 and printed
  # 'ARM U DONE', the queue advanced on it, and the ledger would have recorded silence as a result.
  # Same family as knee-b02 - never let a wrapper's exit code speak for the work.
  local name="$1"; shift
  uv run python -u -m kneemri.train "$@" 2>&1 \
    | tee "$LOGS/$name.full.log" \
    | grep --line-buffered -E "epoch |best |rror|emory"
  local status="${PIPESTATUS[0]}"
  if [ "$status" -ne 0 ]; then
    _mark "$name" "ARM FAILED $name (trainer exit $status) - see $LOGS/$name.full.log"
    return "$status"
  fi
  # A trainer can also exit 0 having produced nothing useful. The best-score line is the artefact the
  # ledger quotes, so its absence is a failure whatever the exit code says.
  if ! grep -q "best holdout" "$LOGS/$name.full.log"; then
    _mark "$name" "ARM FAILED $name (exit 0 but no 'best holdout' line) - see $LOGS/$name.full.log"
    return 1
  fi
  # The completion marker is emitted HERE, on success only, rather than by each arm. An arm that
  # prints its own marker prints it whatever happened - which is how a dead arm advanced the queue.
  _mark "$name" "ARM DONE $name"
}

_mark() {
  # Write an arm's verdict to BOTH the terminal and the arm's own log.
  #
  # It used to go only to stdout, which lands wherever the caller redirected - so an arm's full.log
  # held every epoch and no verdict, and a gate pointed at that log waited forever for a marker that
  # was never going to arrive there. Found while queueing the resolution arm: the GPU would simply
  # have gone idle with nothing reporting an error.
  local name="$1" message="$2"
  echo "$message" | tee -a "$LOGS/$name.full.log"
}

wait_for() {
  # wait_for <completion-marker> <file>. Blocks until the marker appears.
  local marker="$1" file="$2"
  until grep -q "$marker" "$file" 2>/dev/null; do sleep 60; done
}

wait_for_pid() {
  # Block until a process exits. The gate of LAST resort - `wait_for` on a marker is the default.
  #
  # It exists because a marker is written by the arm's shell, not by the trainer, so a trainer that
  # outlives its shell finishes without ever announcing it. That is a real state, not a hypothetical:
  # cutting folds 2-3 out of a running queue meant killing the parent script while deliberately
  # leaving its python child alive, and nothing was ever going to write that child's marker.
  #
  # Gates on the PID rather than the checkpoint file, because torch.save is not atomic - the .pt
  # appears at the first byte written, and a queue that fires there hands the next arm a truncated
  # file.
  local pid="$1"
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
}

wait_for_cache() {
  # Block until a cache is genuinely finished, keyed on its MASK file.
  #
  # The images .npy is allocated at build start and filled over hours, so its existence proves only
  # that a build began - gating on it is how an arm nearly trained on a mostly-zero array. The mask is
  # different: build() writes it with a single np.save AFTER the decode loop returns, so it exists if
  # and only if the build ran to completion.
  #
  # Better than grepping the build log, which is what this replaced: the mask lives on the same
  # filesystem as the artefact it certifies, so it cannot be lost, rotated or truncated separately -
  # and a build killed by the OS leaves no trace in a log at all. On 2026-08-11 the 448 px build was
  # killed at 2200/4407 and its log simply stopped, with no error to grep for.
  local size="$1" slices="${2:-12}"
  until [ -f "$PROCESSED/mask_${size}px_${slices}sl.npy" ]; do sleep 60; done
}
