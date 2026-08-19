# Shared plumbing for the training arms. Source it; do not run it.
#
# An arm is a hypothesis, a control number, and one `kneemri.training.train` invocation. Everything an arm
# needs that is not the hypothesis itself lives here, so an arm file is only ever the interesting part.
#
# Two rules are encoded rather than remembered:
#   - The data root is resolved through `core.paths`, never hardcoded. paths.yaml is gitignored and
#     machine-specific; an arm that spells out D:/data is not runnable on another machine.
#   - A readiness gate keys on a marker the producer writes when it FINISHES. An output path proves a
#     job started and nothing more - `wait_for` exists so that mistake cannot be made per-arm again.

set -u

# The shared recipe, so an arm and the checker see the same flags.
source "$(dirname "${BASH_SOURCE[0]}")/recipe.sh"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PROCESSED="$(uv run python -c 'from pathlib import Path
from core.paths import DataRoot
print(DataRoot.from_config(Path("paths.yaml")).processed().as_posix())')"
LABELS="$PROCESSED/labels/report_labels_v2.csv"
LOGS="$PROCESSED/experiments"
mkdir -p "$LOGS"

hold_card() {
  # Block until this shell owns the GPU, then release it when the arm exits.
  #
  # after.sh serialises the arms WITHIN one chain, and nothing stopped two chains from firing at once.
  # On 2026-08-12 two waiters were armed on the same gate PID; when it exited they both started, and
  # the card went to 31.9 GB of 32.6 with the training arm crawling at 32-37 img/s against ~520 with
  # the card to itself. Not a crash - Windows pages GPU memory over PCIe, so it reports 99%
  # utilisation at a third of the power budget and simply takes fifteen times longer.
  #
  # `mkdir` is the lock because it is ATOMIC: two shells racing cannot both create the same directory,
  # where a test-then-write on a lockfile can interleave. A lock whose holder is gone is cleared rather
  # than waited on, so a killed arm does not wedge the queue forever.
  # ONE ARM AT A TIME, NOT ONE CHAIN AT A TIME. The lock used to be taken here and released by an
  # EXIT trap, which meant a chain of several arms took it once and then blocked FOREVER on its own
  # lock at the second arm: mkdir fails, the recorded holder is this very shell, kill -0 says it is
  # alive, and the loop sleeps against itself. A four-arm chain would have idled the card silently
  # after its first arm. `release_card` after each arm is what makes a chain possible at all.
  #
  # The trap is a safety net for a killed chain, and it must expand the path NOW rather than at exit:
  # written as 'rm -rf "$lock"' the variable is read when the trap fires, by which time the function's
  # local is gone — under `set -u` that is "lock: unbound variable" and the lock is never removed.
  LOCK_DIR="$LOGS/gpu.lock"
  until mkdir "$LOCK_DIR" 2>/dev/null; do
    local holder
    holder="$(cat "$LOCK_DIR/pid" 2>/dev/null)"
    if [ -n "$holder" ] && ! kill -0 "$holder" 2>/dev/null; then
      echo "clearing a GPU lock held by dead process $holder"
      rm -rf "$LOCK_DIR"
      continue
    fi
    sleep 60
  done
  echo $$ > "$LOCK_DIR/pid"
  trap "rm -rf '$LOCK_DIR'" EXIT
  wait_for_free_card
}

ARM_NEEDS_MIB=18432
# The free VRAM an arm needs before it may start. DERIVED, not chosen: our arms log "peak memory
# 16.24 GB reserved, 15.03 GB allocated", and the driver context plus cuDNN's kernel workspaces sit ON
# TOP of the caching allocator's reservation. 18 GB is the smallest figure that holds the measured peak
# with that headroom, and it leaves 14 GB of a 32.6 GB card for anyone else.

FOREIGN_WAITS=60
# How many minutes to wait out a foreign tenant before starting anyway, loudly. Unbounded would let a
# process that parks 20 GB and never exits idle our whole queue in silence; an hour is longer than any
# arm we run, so a real trainer on the other side clears well within it.

wait_for_free_card() {
  # Wait out a CUDA process this lock cannot see — one belonging to ANOTHER REPOSITORY.
  #
  # The mkdir lock above serialises kneemri arms against each other and is blind to everything else. On
  # 2026-08-19 a second project's trainer took the card to 31949 MiB of 32607 while one of our arms was
  # mid-epoch; Windows pages GPU memory over PCIe rather than failing, so the arm kept running at 30-170
  # img/s against its own peak of 1090 for the rest of the run. `nvidia-smi` reported 99% utilisation
  # throughout — at 183 W of a 575 W limit. Power draw is the tell; utilisation cannot see this at all.
  #
  # PREVENTION IS SEPARATE FROM DETECTION and both are needed. `kneemri.training.heartbeat` reports a
  # collapse against the run's own peak, which is what saves an arm already underway. This stops the arm
  # from starting into the overflow in the first place, which the heartbeat cannot do.
  #
  # NOT `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which is the obvious reach and is dead here:
  # torch prints "expandable_segments not supported on this platform" at startup on this machine, and
  # three arms carried the flag without one of them being helped by it. Nor `--vram-gb`, which caps our
  # allocator rather than the device — a cap only converts thrashing into a clean OOM when it leaves the
  # driver room, and set above what the card can spare it PERMITS the thrashing it was meant to prevent.
  local free waited=0
  while :; do
    free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
    # No card, no driver, or a query that failed: this is a serialisation gate, not a hardware check.
    # Refusing to run because we could not measure would turn a missing tool into a stalled queue.
    case "$free" in ''|*[!0-9]*) return 0 ;; esac
    [ "$free" -ge "$ARM_NEEDS_MIB" ] && return 0
    if [ "$waited" -ge "$FOREIGN_WAITS" ]; then
      echo "STARTING ANYWAY after ${waited}m: only ${free} MiB free, want ${ARM_NEEDS_MIB}. Expect PCIe paging" >&2
      return 0
    fi
    [ "$waited" -eq 0 ] && echo "waiting for another process to release the card: ${free} MiB free, want ${ARM_NEEDS_MIB}"
    waited=$((waited + 1))
    sleep 60
  done
}

release_card() {
  # Hand the card to the next chain the moment THIS arm is done, rather than at the end of ours.
  # An `if` rather than `[ ... ] && rm`, because that construct returns non-zero when the guard fails
  # and this file runs under `set -e` — releasing a lock we do not hold would then kill the chain.
  if [ -n "${LOCK_DIR:-}" ]; then
    rm -rf "$LOCK_DIR"
  fi
  LOCK_DIR=""
}

run_snapshot() {
  # Run an arm from a COPY, never from the file in the repo.
  #
  # bash reads a script incrementally, by byte offset, as it executes. Editing that file mid-run makes
  # it resume at a stale offset - on 2026-08-11 a sweep that rewrote every arm's trailing marker
  # truncated whole_stack.sh under a running bash, which died with "unexpected EOF while looking for
  # matching quote" and took a 2-hour arm with it. The repo has to stay editable while long work runs,
  # so the run gets its own immutable copy.
  local arm="$1"; shift
  local name; name="$(basename "$arm" .sh)"
  local snapshot="$LOGS/$name.running.sh"
  cp "$arm" "$snapshot"

  # THE PACKAGE IS SNAPSHOT TOO, and it was not: only the arm script was copied, while the trainer
  # imported kneemri/ live from the repo. On 2026-08-12 speed_combo launched during an edit and died
  # with `NameError: name 'Encoder' is not defined` — a half-written import, three seconds wide, that
  # cost the whole arm. The repo has to stay editable while long work runs; that is the entire reason
  # the arm gets a copy, and the package is no different.
  # PYTHONPATH IS NOT ENOUGH, checked rather than assumed: python puts the working directory ahead of
  # it, so a trainer launched from the repo imports the repo whatever PYTHONPATH says. The trainer
  # therefore RUNS from the snapshot directory, which needs paths.yaml beside the packages.
  ARM_SRC="$LOGS/$name.src"
  export ARM_SRC
  rm -rf "$ARM_SRC"; mkdir -p "$ARM_SRC"
  cp -r "$REPO/kneemri" "$REPO/core" "$ARM_SRC/"
  cp "$REPO/paths.yaml" "$ARM_SRC/"
  # Wait for the card BEFORE the arm starts, so a second chain queues instead of contending.
  hold_card
  # "$@" is forwarded: an arm that takes arguments (which folds, which resolution) was being run with
  # none, so its loop iterated zero times and it exited 0 with an empty log. A silent no-op is the
  # worst failure available - the GPU simply sat idle and nothing reported anything.
  local status=0
  ARM_LIB="$REPO/experiments/lib.sh" bash "$snapshot" "$@" || status=$?
  release_card
  return "$status"
}

retired() {
  # An arm whose GATE refuted it: the hypothesis stays readable, the run is refused.
  #
  # A refuted arm is not deleted — the file is the record of what was tried and why it lost, and the
  # ledger links to it. But it must not be runnable: multiscale.sh asks for a 71.6 GB cache and a
  # training slot for a build that crop80_tissue priced at about +0.0016 on the mean, and nothing in
  # the file stopped a queue from picking it up.
  #
  # Exit 1, not 0. A retired arm reached by a queue is a mistake the queue should report; an arm that
  # silently does nothing is the worst outcome available here, which is the same reason run_train
  # reads PIPESTATUS rather than trusting a contented grep.
  echo "RETIRED ARM: $1"
  exit 1
}

require() {
  # Fail before spending GPU, not after. Every path an arm depends on goes through here.
  for path in "$@"; do
    [ -e "$path" ] || { echo "MISSING: $path"; exit 1; }
  done
}

run_train() {
  # $1 is the arm's name; the rest are kneemri.training.train flags.
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
  # FROM THE SNAPSHOT, never the repo: the package is copied per arm so an edit mid-run cannot reach a
  # trainer that is already going. `cd` rather than PYTHONPATH, because the working directory wins.
  # `--project "$REPO"` is what makes the snapshot survivable: `uv run` resolves its ENVIRONMENT from
  # the working directory, and the snapshot holds only the packages — no pyproject.toml — so running
  # there gave a bare interpreter and `ModuleNotFoundError: jaxtyping` in under a second. The project
  # comes from the repo, the IMPORTS come from the snapshot (python puts the working directory first),
  # which is the split this needed all along.
  # --no-sync, because a gate or a git push run while this trains will otherwise RE-SYNC .venv under
  # the interpreter and kill it. Measured 2026-08-13: pyproject.toml was edited at 11:28, a push at
  # 14:03 ran the pre-push hook's `uv run pytest`, uv rewrote uv.lock at 14:05, and the arm died mid
  # epoch 4 with exit 127 and no traceback. run_snapshot protects the SOURCE from edits mid-run; this
  # protects the ENVIRONMENT, which is the other half and was missing.
  (cd "${ARM_SRC:-$REPO}" && uv run --no-sync --project "$REPO" python -u -m kneemri.training.train "$@") 2>&1 \
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
  # `kill -0` ALONE IS NOT A LIVENESS TEST ON THIS MACHINE. It only knows processes this bash session
  # can signal, and a trainer launched from PowerShell — or from another bash — is a native Windows PID
  # it has never heard of. It then reports that live process as DEAD, and the gate fails OPEN: on
  # 2026-08-16 an arm queued behind a running final fit started immediately, five seconds later, onto a
  # card holding 26 GB. `tasklist` asks the OS instead, and is the answer that is actually true here.
  local pid="$1"
  if ! _alive "$pid"; then
    echo "WARNING: pid $pid is not running — either it already finished, or the number is wrong"
  fi
  while _alive "$pid"; do sleep 60; done
}

_alive() {
  # Does this PID name a running process? Either answer counts; only "no" from BOTH means gone.
  local pid="$1"
  kill -0 "$pid" 2>/dev/null && return 0
  # `//FI`, not `/FI`: git-bash rewrites a leading slash as a path and hands tasklist
  # "C:/Program Files/Git/FI", which errors out — and an erroring liveness probe says "dead".
  tasklist //FI "PID eq $pid" //NH 2>/dev/null | grep -qE "[[:space:]]$pid[[:space:]]"
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
  # THE KEY IS ASKED FOR, never spelled here. This used to rebuild it in bash from size, slices and
  # scheme, which covers one naming axis of four: an arm waiting on 336px_12sl_80mm_tissue watched
  # mask_336px_12sl.npy - the canonical mask, which already exists - and proceeded instantly against a
  # cache nobody had built. Reporting ready for the wrong file is the failure this function exists to
  # prevent, so the reading goes to the builder that names it, in full.
  local size="$1" slices="${2:-12}" scheme="${3:-canonical}" crop="${4:-130}" centre="${5:-image}"
  local key
  key="$(uv run python -m kneemri.corpus.build_cache --print-key     --image-size "$size" --slices "$slices" --scheme "$scheme" --crop-mm "$crop" --centre "$centre")"
  [ -n "$key" ] || { echo "wait_for_cache: the builder named no key"; return 1; }
  until [ -f "$PROCESSED/mask_${key}.npy" ]; do sleep 60; done
}
