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

# WHICH lib.sh THIS SHELL IS RUNNING, captured where it can still be true: at source time.
#
# bash parses a sourced file ONCE, into function definitions held in this shell's memory. A chain is
# one long-lived shell that sources this file at launch and then runs arm after arm, so every arm in a
# chain runs the lib.sh that existed when the CHAIN started - not when the ARM started. run_snapshot
# snapshots the arm and the package and points ARM_LIB at the LIVE file, which helps an editor and does
# nothing for a chain already parsed.
#
# On 2026-08-19 that cost an hour: `_seat_value` was committed at 12:36, two arms launched from chains
# started before it landed, and both finished with no SEAT VALUE line. The mechanism looked broken. It
# was absent, which is a different bug with a different fix, and nothing in either log said so.
#
# The blob hash rather than HEAD: it is CONTENT, so an uncommitted edit reads differently from the
# commit it will become, which is the case a chain is most likely to be stale against.
# The file this shell parsed, named once so the blob check cannot drift from the thing it checks.
LIB_SOURCE="${BASH_SOURCE[0]}"

# The CONTENT hash of that file as the working tree holds it right now, and whether this shell is
# behind it. Separated from the stamp because run_train has to ask the question again per arm.
_lib_blob() { git -C "$REPO" hash-object "$LIB_SOURCE" 2>/dev/null || echo unknown; }

_lib_is_stale() { [ "$(_lib_blob)" != "$LIB_PARSED" ]; }

LIB_PARSED="$(_lib_blob)"
LIB_PARSED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

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
  # A PARKED ARM MUST SAY IT IS PARKED. This loop used to wait in total silence, so a queued arm's log
  # sat at zero bytes whether it was blocked on a live sibling or had died at launch — and those are
  # opposite states: one keeps the card busy next, the other leaves it idle. On 2026-08-19 a parked
  # waiter died and the 0-byte log said nothing, so the idle was found only after the running arm
  # finished. The heartbeat makes the distinction readable without asking the process table: an empty
  # log now means DEAD, and a waiting arm names who it is waiting for and for how long.
  local blocked=0
  until mkdir "$LOCK_DIR" 2>/dev/null; do
    local holder
    holder="$(cat "$LOCK_DIR/pid" 2>/dev/null)"
    if [ -n "$holder" ] && ! kill -0 "$holder" 2>/dev/null; then
      echo "clearing a GPU lock held by dead process $holder"
      rm -rf "$LOCK_DIR"
      continue
    fi
    [ $((blocked % WAIT_HEARTBEAT_MINUTES)) -eq 0 ] &&
      echo "parked ${blocked}m behind arm ${holder:-unknown} — the card is held, this shell is alive"
    blocked=$((blocked + 1))
    sleep 60
  done
  [ "$blocked" -gt 0 ] && echo "took the card after ${blocked}m parked"
  echo $$ > "$LOCK_DIR/pid"
  trap drop_own_card EXIT
  wait_for_free_card
}

drop_own_card() {
  # Remove the lock ONLY if this shell is still the recorded holder.
  #
  # The trap used to be `rm -rf '<path>'`, which deletes whatever lock exists at exit — including one
  # ANOTHER CHAIN IS HOLDING. Measured 2026-08-19: a shell that took the card, released it after its
  # arm, and exited later wiped a lock a second chain had taken in between, leaving that chain running
  # an arm with the lock gone and a third free to start on top of it. Two arms on one card is the
  # 31.9-GB overflow that pages over PCIe at a tenth of the throughput, and the serialiser was the thing
  # producing it. The empty gpu.lock observed at 06:40 with an arm mid-epoch was this.
  #
  # An ownership test rather than `trap - EXIT` in release_card, because that only fixes the shells that
  # remember to release; this also covers a killed arm, which is the case the trap exists for.
  local dir="$LOGS/gpu.lock"
  [ "$(cat "$dir/pid" 2>/dev/null)" = "$$" ] && rm -rf "$dir"
  return 0
}

ARM_NEEDS_MIB=18432
# The free VRAM an arm needs before it may start. DERIVED, not chosen: our arms log "peak memory
# 16.24 GB reserved, 15.03 GB allocated", and the driver context plus cuDNN's kernel workspaces sit ON
# TOP of the caching allocator's reservation. 18 GB is the smallest figure that holds the measured peak
# with that headroom, and it leaves 14 GB of a 32.6 GB card for anyone else.

WAIT_HEARTBEAT_MINUTES=5
# How often a parked arm repeats that it is still parked. Five minutes is short against the ~45-minute
# arms it waits behind and long enough that an hour of parking costs twelve lines, not sixty.

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
    drop_own_card
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
  # THE CARD IS NOT TAKEN HERE. It is taken by `run_train`, which is the one point EVERY path that
  # spends GPU crosses — see the note there. Gating the snapshot instead left the documented smoke
  # procedure ungated, and copying the package spends no GPU, so there is nothing to protect until the
  # trainer starts.
  # "$@" is forwarded: an arm that takes arguments (which folds, which resolution) was being run with
  # none, so its loop iterated zero times and it exited 0 with an empty log. A silent no-op is the
  # worst failure available - the GPU simply sat idle and nothing reported anything.
  local status=0
  ARM_LIB="$REPO/experiments/lib.sh" bash "$snapshot" "$@" || status=$?
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
  # THE CARD IS TAKEN HERE, because this is the ONE point every path that spends GPU crosses.
  #
  # It used to be taken in `run_snapshot`, which covers a queued arm and nothing else. The smoke
  # procedure CLAUDE.md documents — source the lib, `run_train ... --studies 60 --epochs 1` — reaches
  # the trainer without passing through run_snapshot, so it was ungated by construction, and so is any
  # ad-hoc run. On 2026-08-19 a smoke launched that way onto a card already running an arm and died with
  # `CUDA out of memory ... 4.84 GiB is free`. The incumbent survived because the NEWCOMER lost the
  # allocation race, which is luck, not a design: the same collision at a smaller footprint is two arms
  # sharing a card, which is the PCIe-paging collapse the lock exists to prevent.
  #
  # Per call rather than per arm, and released before returning: a sweep arm that calls this several
  # times now lets another chain interleave between its runs instead of holding the card across all of
  # them.
  # STALENESS IS FIXED HERE, not merely reported. LIB_PARSED made a chain's staleness legible (knee-yn6j
  # option b); this is option (a), and it is what stops the arm in front of us from running the wrong
  # lib. The re-source re-parses every function INCLUDING this one, and bash resolves a function at CALL
  # time, so the recursive call below is the NEW run_train — the current arm gets the current mechanisms
  # rather than the next arm getting them. `_LIB_REFRESHED` bounds it to one attempt, because a repo
  # without git answers `unknown` forever and a loop here would spend the slot instead of the arm.
  #
  # queue.sh needs none of this: it runs each arm as `bash experiments/arms/<arm>.sh`, a fresh shell
  # that sources this file fresh. The stale case is the hand-typed chain that sources once and then
  # calls run_train arm after arm, which is how two arms finished with no SEAT VALUE line on 2026-08-19.
  if _lib_is_stale && [ -z "${_LIB_REFRESHING:-}" ]; then
    echo "run_train: this shell holds lib.sh $LIB_PARSED, the tree has $(_lib_blob) - re-sourcing before $1"
    _LIB_REFRESHING=1
    source "${BASH_SOURCE[0]}"
    run_train "$@"
    local refreshed=$?
    # Cleared rather than left set: the guard bounds ONE call, so the arm after this one is checked
    # again against whatever the tree holds by then.
    unset _LIB_REFRESHING
    return "$refreshed"
  fi
  hold_card
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
  # Released on EVERY path out, including the failures. A crashed trainer that kept the lock would idle
  # the queue behind it for as long as the chain lived, which is the failure the marker checks below
  # exist to make visible rather than one to add.
  release_card
  # The provenance goes to the log on EVERY path out, before the verdict, so a missing mechanism is
  # attributable in one grep instead of an hour. It cannot be written any earlier: `tee` without -a
  # truncates, so anything stamped before the trainer runs is erased by the trainer's first line.
  _mark "$name" "LIB $LIB_PARSED parsed $LIB_PARSED_AT (experiments/lib.sh as this shell holds it)"
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
  _seat_value "$name"
}

_seat_value() {
  # The SECOND ruler, printed beside the first rather than remembered.
  #
  # Every arm header names a matched control - the same recipe with one field changed - which answers
  # "did this knob do anything" and not "does this earn a seat". The two came apart once the roster
  # stopped being the control's family: res392_ws scores 0.8523 alone, shipped in the four that scored
  # 0.925, and a greedy over 22 members takes it NINTH, the first pick that LOWERS the mix. Solo score
  # ranked it into a submission; marginal contribution never would have.
  #
  # Here rather than in a convention, because a convention is a thing to forget at exactly the moment
  # a number looks good. Both readings now land in the same log, at the same time, unasked.
  #
  # THE CHECKPOINT COMES FROM THE TRAINER'S OWN BEST LINE, not from the arm name. The kept epoch's
  # file is the artefact the ledger quotes and the only one whose record matches the reported score;
  # rebuilding the path from "$name_s2026.pt" would be a second naming rule to drift from the first.
  #
  # A FAILURE HERE IS REPORTED, NEVER PROPAGATED. The read joins on the shared holdout, so an arm on
  # another fold or a smoke split raises rather than scores - correctly, that is `shared`'s refusal.
  # But the arm TRAINED, and marking a finished arm failed because an after-the-fact reading could not
  # join would throw away forty minutes of GPU over arithmetic that costs none.
  local name="$1" best checkpoint
  best="$(grep -m1 'best holdout' "$LOGS/$name.full.log")"
  checkpoint="${best##*-> }"
  checkpoint="${checkpoint%% |*}"
  if [ ! -f "$checkpoint" ]; then
    _mark "$name" "SEAT VALUE skipped: no checkpoint at '$checkpoint'"
    return 0
  fi
  # --no-sync for the same reason run_train uses it: another chain may already be training, and a
  # re-sync of .venv under its interpreter kills it mid-epoch.
  local report
  if ! report="$(uv run --no-sync --project "$REPO" python -m kneemri.kaggle.seat_value \
      --candidates "$checkpoint" 2>&1)"; then
    _mark "$name" "SEAT VALUE unavailable for $name: $report"
    return 0
  fi
  _mark "$name" "$report"
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
