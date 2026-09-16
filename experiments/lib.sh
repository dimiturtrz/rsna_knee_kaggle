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

# AN ARM RUNS IN ITS OWN ENVIRONMENT, and this is not a preference.
#
# `uv` syncs .venv to EXACTLY the extras a command asks for, in place, with no lock. Every pre-commit
# hook alternates `--extra dev` and `--extra devtools`, so a commit made while an arm is fitting
# rewrites the interpreter that arm is executing. The arm dies with exit 127 and NO traceback, which
# reads as a crashed run rather than as a rewritten venv - it cost three arms before it was named.
#
# THE CONVERSE IS FALSE, and this sentence is here because its absence cost a fourth. Exit 127 does NOT
# imply a rewritten venv. It is the shell's code for `command not found` AND what it holds after a
# native abort, so a CUDA driver fault presents identically. Read the exception line the failure marker
# now carries; only a bare interpreter error (`ModuleNotFoundError`, or no traceback at all) is this.
#
# .venv-arm is a second, isolated environment nothing else ever syncs. Pointing UV_PROJECT_ENVIRONMENT
# at it makes the isolation the DEFAULT for every arm instead of a flag each one has to remember.
export UV_PROJECT_ENVIRONMENT="$REPO/.venv-arm"
[ -d "$UV_PROJECT_ENVIRONMENT" ] || uv sync --extra dev

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

# Both roots out of ONE interpreter start: a second `uv run` here costs every arm a fresh import of
# torch's dependency tree before it has read a single study.
_lib_roots="$(uv run --no-sync python -c 'from pathlib import Path
from core.paths import DataRoot
root = DataRoot.from_config(Path("paths.yaml"))
print(root.processed().as_posix())
print(root.models().as_posix())')"
PROCESSED="$(echo "$_lib_roots" | sed -n 1p)"
MODELS="$(echo "$_lib_roots" | sed -n 2p)"
LABELS="$PROCESSED/labels/report_labels_v2.csv"
LOGS="$MODELS/experiments"

# Touch this to stop a chain STARTING further arms; the running one finishes. See run_train.
# In the repo's gitignored logs/ rather than under $PROCESSED, because the request comes from a person
# or a peer session and it has to be findable without knowing where paths.yaml points.
YIELD_FILE="$REPO/logs/YIELD_CARD"
mkdir -p "$LOGS"

# The GPU lock is MACHINE-scoped, not project-scoped, because the card is.
#
# It used to live at "$LOGS/gpu.lock" -- inside this project's data root, where no other project can
# see it. That serialises our own arms against each other and nothing else. Measured 2026-08-21:
# band140 reserved 28934 MiB of the 32607 on the card while a sibling project's CUDA eval grew to
# 11.4 GB, and free VRAM fell to 1923 MiB. Our arm sat in the free-VRAM gate for the better part of an
# hour with a lock it held uncontested, because the process it was actually contending with was
# holding a different lock in a directory ours does not exist in.
#
# $HOME rather than a data root: it is the one path both projects resolve identically without either
# reading the other's config, and it is outside both repos. GPU_LOCK_DIR overrides it for a machine
# where that is not true.
GPU_LOCK_DIR="${GPU_LOCK_DIR:-$HOME/.gpu.lock}"

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
  LOCK_DIR="$GPU_LOCK_DIR"
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
  trap 'drop_claim; drop_own_card' EXIT
  wait_out_legacy_lock
  wait_out_peer_claim
  wait_for_free_card "$@"
  claim_card "$@"
  # Explicit, because this function's status is whatever its last command left behind and an arm runs
  # under `set -e`. wait_for_free_card ends on a test in one of its branches.
  return 0
}

wait_out_legacy_lock() {
  # TRANSITION GUARD, removable once no arm snapshotted before 2026-08-21 can still be running.
  #
  # An arm executes from a SNAPSHOT of this lib taken when it was launched, so moving the lock does not
  # move it for anything already in flight: band140 was mid-epoch under "$LOGS/gpu.lock" when this
  # landed, and the next arm in its chain snapshots the NEW lib and takes a lock the running arm has
  # never heard of. Two arms on one card is exactly the 31.9-GB PCIe-paging overflow the lock exists to
  # prevent, and the move would have produced it on its first use. The new lock is already held by the
  # time this runs, so waiting here cannot deadlock -- nothing old looks at the new path.
  local legacy="$LOGS/gpu.lock" holder waited=0
  while [ -d "$legacy" ]; do
    holder="$(cat "$legacy/pid" 2>/dev/null)"
    [ -n "$holder" ] && kill -0 "$holder" 2>/dev/null || { rm -rf "$legacy"; break; }
    [ "$waited" -eq 0 ] && echo "waiting out a pre-move lock held by $holder at $legacy"
    waited=$((waited + 1))
    sleep 60
  done
  return 0
}

CLAIM_FILE="$REPO/logs/CARD_CLAIM"
PEER_CLAIMS="${PEER_CLAIMS:-$REPO/../*/logs/CARD_CLAIM}"
# An INTENT file, which is the one thing neither the lock nor the free-VRAM gate can represent.
#
# Both of those read the card's CURRENT state, so both are blind in the same direction: on 2026-08-24 the
# gate correctly saw ~30 GB free at 09:22Z and let a 72-minute arm through, and a sibling project's ~4.8 GB
# job arrived ninety seconds later. Total hit 31.8 GB of 32.6 and the arm reported 94 img/s against its own
# peak of 1221 — a 13x paging collapse that would have made 72 minutes into fifteen hours. Nothing was
# broken. Both sides asked the right question and neither could see a job that had not started yet.
#
# NOT a second lock. The sibling took the protocol and implemented it as a claim FILE rather than the
# `$HOME/.gpu.lock` mkdir we already had, so the machine now has two mechanisms that each cover one
# direction. Rather than argue the sibling into ours, this reads theirs — a claim is advisory and a reader
# costs nothing, where a lock nobody else takes is a lock that protects nobody. We write ours in the same
# format for exactly that reason.

claim_card() {
  # State the intent this arm is about to act on, so a peer deciding at the same moment can see it.
  local finish
  finish="$(date -u -d "+${CLAIM_HOURS:-2} hours" +%Y-%m-%dT%H:%MZ 2>/dev/null)"
  mkdir -p "$(dirname "$CLAIM_FILE")"
  {
    echo "CLAIM $(date -u +%Y-%m-%dT%H:%MZ)"
    echo "holder: kneemri (pid $$)"
    echo "job: ${CLAIM_JOB:-unnamed arm}"
    echo "footprint_mib: $(arm_needs_mib "$@")"
    echo "expected_finish: ${finish:-unknown}"
  } > "$CLAIM_FILE"
  return 0
}

drop_claim() {
  # Only ours, and only if it is still ours — the same ownership test the lock trap learned on 2026-08-19.
  [ "$(sed -n 's/^holder: kneemri (pid \(.*\))$/\1/p' "$CLAIM_FILE" 2>/dev/null)" = "$$" ] && rm -f "$CLAIM_FILE"
  return 0
}

wait_out_peer_claim() {
  # Park while a sibling project has said it is about to use the card, up to its own stated finish time.
  #
  # BOUNDED BY THE CLAIM'S OWN EXPIRY, not by a timeout of ours. A claim file is written by a process that
  # can be killed, and a dead holder's file does not vanish — the lock has the same failure and clears it
  # by asking whether the pid is alive, which does not work across projects. The stated finish is what the
  # writer offered as the promise, so a claim past it is stale by its author's own account and is ignored
  # LOUDLY rather than silently: a stale advisory file must not be able to idle our queue forever.
  local claim now finish waited=0
  while :; do
    claim=""
    for candidate in $PEER_CLAIMS; do
      # WHOSE claim it is, read off the file, rather than which PATH it sits at. The glob expands to
      # `$REPO/../*/logs/CARD_CLAIM`, which matches our own repo through `..` and spells it differently
      # from `$CLAIM_FILE` — so the path test never fired, our own claim was picked as the peer, its
      # kneemri-format `holder` line still parsed, and the loop then `break`s on the FIRST candidate.
      # A leftover claim of ours therefore made this whole function return before it could see a real
      # peer claim listed after it: the guard was disabled by the thing it was guarding. Measured
      # 2026-08-24T12:10Z, where it announced "IGNORING a peer claim" about a file we had just written.
      grep -q '^holder: kneemri' "$candidate" 2>/dev/null && continue
      [ -f "$candidate" ] && claim="$candidate" && break
    done
    [ -z "$claim" ] && { [ "$waited" -gt 0 ] && echo "peer released the card after ${waited}m parked"; return 0; }
    now="$(date -u +%Y-%m-%dT%H:%MZ)"
    finish="$(sed -n 's/^expected_finish: *//p' "$claim" | head -1)"
    # String comparison rather than epoch arithmetic: ISO-8601 UTC sorts lexicographically by construction,
    # and `date -d` on an arbitrary field is a parse this loop should not be able to fail on.
    case "$finish" in
      20[0-9][0-9]-*Z) [ "$now" \> "$finish" ] && {
        echo "IGNORING a peer claim at $claim: it promised $finish and it is now $now" >&2
        return 0
      } ;;
      *) echo "IGNORING a peer claim at $claim: no readable expected_finish, so it cannot bound our wait" >&2
         return 0 ;;
    esac
    [ "$waited" -eq 0 ] && { echo "parking behind a peer claim until $finish:"; sed 's/^/  /' "$claim"; }
    [ $((waited % WAIT_HEARTBEAT_MINUTES)) -eq 0 ] && [ "$waited" -gt 0 ] &&
      echo "parked ${waited}m behind the peer claim, which runs to $finish"
    waited=$((waited + 1))
    sleep 60
  done
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
  local dir="$GPU_LOCK_DIR"
  [ "$(cat "$dir/pid" 2>/dev/null)" = "$$" ] && rm -rf "$dir"
  return 0
}

ARM_NEEDS_MIB=18432
# The free VRAM an arm needs before it may start, WHEN ITS OWN FLAGS CANNOT BE READ. Derived, not
# chosen: our arms log "peak memory 16.24 GB reserved, 15.03 GB allocated", and the driver context plus
# cuDNN's kernel workspaces sit ON TOP of the caching allocator's reservation. 18 GB is the smallest
# figure that holds THAT arm's measured peak with that headroom.
#
# IT USED TO BE THE ONLY ANSWER, AND IT WAS WRONG IN BOTH DIRECTIONS. A 12-slice arm's number handed to
# every arm is the same fitted-on-one-configuration failure `DeviceBudget` has recorded five times, one
# layer out where the class could not see it. It OVERSTATES a small recipe — `encoder_adapt_thawed`
# reserves 9.6 GB at batch 2 and parked behind an 18 GB gate on 2026-08-21 with 15.8 GB of the card
# free, against an hour-long wait bound. And it UNDERSTATES the shipped one: the settled 20-slice recipe
# derives 28.3 GB, so those arms were cleared to start with ten gigabytes less free than they go on to
# reserve, which is precisely the PCIe paging this gate exists to prevent.
#
# THE STRICTER GATE IS A REAL CHANGE IN THE QUEUE'S CHARACTER and is named rather than slipped in. A
# 20-slice arm now wants 28.3 GB of a 32.6 GB card, which a co-tenant of any size denies — so those arms
# will reach `FOREIGN_WAITS` and start anyway, loudly, where before they started immediately and
# silently. That is the honest reading of the physics either way; what changes is that the log now says
# the arm is starting into contention instead of implying it fitted.

arm_needs_mib() {
  # What THIS recipe reserves, from the same cost model the trainer caps itself with — one home for it.
  #
  # `--no-sync` for the reason run_train names: a gate or a push running alongside must not re-sync the
  # environment under a live arm. A failed derivation (an unparseable flag set, a missing interpreter,
  # or a recipe whose slices are unknown, which `DeviceBudget.needed` answers with 0) falls back to the
  # constant rather than to zero — a gate that cannot measure must not become a gate that permits.
  local derived
  derived="$(cd "$REPO" && uv run --no-sync python -m kneemri.experiments.vram_need "$@" 2>/dev/null | tail -1)"
  case "$derived" in ''|0|*[!0-9]*) echo "$ARM_NEEDS_MIB" ;; *) echo "$derived" ;; esac
}

WAIT_HEARTBEAT_MINUTES=5
# How often a parked arm repeats that it is still parked. Five minutes is short against the ~45-minute
# arms it waits behind and long enough that an hour of parking costs twelve lines, not sixty.

FOREIGN_WAITS=${FOREIGN_WAITS:-60}
# How many minutes to wait out a foreign tenant before starting anyway, loudly. Unbounded would let a
# process that parks 20 GB and never exits idle our whole queue in silence; an hour is longer than any
# arm we run, so a real trainer on the other side clears well within it.

compute_tenants() {
  # The pids COMPUTING on this card, read off the driver's plain report — one line per holder, empty
  # when the card is idle.
  #
  # NOT `--query-compute-apps`, which on this display driver enumerates every client of the device
  # (thirty-one of them, Explorer and two Firefoxes included) and answers `[N/A]` for every one of their
  # memory figures. The `Type` column that separates `C` from `C+G` exists in no CSV query, so this
  # parses the human table for it, exactly as `kneemri.experiments.card_check` does and for the reason
  # recorded there.
  #
  # The pid is matched as the last numeric field BEFORE the type rather than by column index, because a
  # card without MIG prints `N/A` for `GI`/`CI` and one with it prints numbers, which moves every index.
  nvidia-smi 2>/dev/null | awk '
    { gsub(/\|/, " ") }
    { for (i = 1; i < NF; i++) if ($i ~ /^[0-9]+$/ && $(i + 1) == "C") { print $i; break } }'
}

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
  local needs
  needs="$(arm_needs_mib "$@")"
  # SAY THE NUMBER even when the arm passes, or the gate is only legible when it blocks — and the whole
  # defect being fixed here was a figure nobody had to read because it never appeared in a passing log.
  echo "gate: this recipe reserves ${needs} MiB"
  while :; do
    free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
    # No card, no driver, or a query that failed: this is a serialisation gate, not a hardware check.
    # Refusing to run because we could not measure would turn a missing tool into a stalled queue.
    case "$free" in ''|*[!0-9]*) return 0 ;; esac
    [ "$free" -ge "$needs" ] && return 0
    # NOBODY IS COMPUTING, so there is nothing to wait for and the shortfall is the display driver.
    #
    # MEASURED 2026-08-26, on a card granted to us for the whole day: `free` read 29981 MiB against a
    # 20-slice recipe's derived 29992 — ELEVEN MiB short — with no compute process on the device at all.
    # This card keeps ~2.6 GB in window-manager and browser clients that appear in `memory.free` and
    # never release it, so for the shipped recipe the free-VRAM test cannot pass on an idle card. The
    # arm would have parked the full FOREIGN_WAITS hour and then started anyway, into a card that was
    # free the whole time: an hour of the grant spent, and no paging prevented, because the thing this
    # gate exists to wait out was not there.
    #
    # THE TENANT TEST IS THE ONE THAT MATCHES THE MECHANISM. PCIe paging comes from another TRAINER
    # holding gigabytes it will not give back, not from a compositor's framebuffer — a co-tenant is a
    # process, and the driver names them. Same `Type` column, matched `C` exactly and never as a prefix,
    # that `kneemri.experiments.card_check` documents at length: `C+G` is Explorer and Firefox, and
    # reading it loosely reports twenty-four holders of an empty card.
    if [ -z "$(compute_tenants)" ]; then
      [ "$waited" -gt 0 ] && echo "card is free after ${waited}m: no process computes on it (${free} MiB free)"
      return 0
    fi
    if [ "$waited" -ge "$FOREIGN_WAITS" ]; then
      echo "STARTING ANYWAY after ${waited}m: only ${free} MiB free, want ${needs}. Expect PCIe paging" >&2
      return 0
    fi
    [ "$waited" -eq 0 ] && echo "waiting for another process to release the card: ${free} MiB free, want ${needs}"
    waited=$((waited + 1))
    sleep 60
  done
}

release_card() {
  # Hand the card to the next chain the moment THIS arm is done, rather than at the end of ours.
  # An `if` rather than `[ ... ] && rm`, because that construct returns non-zero when the guard fails
  # and this file runs under `set -e` — releasing a lock we do not hold would then kill the chain.
  if [ -n "${LOCK_DIR:-}" ]; then
    drop_claim
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
  # A `--config experiments/recipes/<name>.toml` resolves against the working directory, which is this
  # one, and the recipe is frozen with the package for the same reason the package is.
  if [ -d "$REPO/experiments/recipes" ]; then
    mkdir -p "$ARM_SRC/experiments" && cp -r "$REPO/experiments/recipes" "$ARM_SRC/experiments/"
  fi
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

# AN ARM SOURCING THIS FILE FROM THE REPO IS SNAPSHOTTING ITSELF, because asking every caller to
# remember run_snapshot did not work and cannot.
#
# run_snapshot has existed since 2026-08-12 and every arm carries the comment naming it, yet
# depth_regions_span ran unsnapshotted on 2026-08-19 and died mid-run: editing its header shifted the
# byte offsets bash reads a script by, execution resumed inside a comment, and the arm re-entered
# run_train for a second training run on a card it already held. The log named the REPO path, which is
# how we know it was never copied. Two invocation paths bypassed the copy — queue.sh runs
# `bash experiments/arms/<arm>.sh` directly, and so does a hand-launched arm, which is how most arms
# start. after.sh was the only caller that remembered.
#
# THE GUARD GOES HERE, not in those callers, because `source lib.sh` is the one line every arm has and
# every path crosses — the same argument that puts the card in run_train rather than in run_snapshot.
# A new caller invented tomorrow inherits it for free.
#
# ARM_LIB IS THE SIGNAL, and it is exact rather than a heuristic: run_snapshot sets it when it hands the
# copy to bash, and arm_check sets it to the stub. Unset means nobody has copied this arm. The $0 test
# keeps after.sh and queue.sh themselves — which source this file with $0 pointing elsewhere — out of
# the re-exec entirely.
case "${ARM_LIB:-}" in
  "")
    case "$(cd "$(dirname "$0")" && pwd)/$(basename "$0")" in
      "$REPO/experiments/arms/"*.sh)
        echo "lib.sh: $(basename "$0") was launched from the repo tree; re-running it from a snapshot"
        run_snapshot "$0" "$@"
        exit $?
        ;;
    esac
    ;;
esac

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

_out_path() {
  # The value the arm passed to --out, or nothing. Parsed here rather than assumed from the arm's name:
  # several arms name a checkpoint that has no relation to their invocation name.
  local previous=""
  for flag in "$@"; do
    [ "$previous" = "--out" ] && { echo "$flag"; return 0; }
    previous="$flag"
  done
}

run_pretrain() {
  # $1 is the log tag; the rest are kneemri.training.pretrain flags. The ONE seam an arm may reach the
  # pretrainer through, for the same reason run_train is the one seam to the trainer.
  #
  # THE ABSOLUTE INTERPRETER PATH IS THE DEFECT THIS EXISTS TO CLOSE. `stub_lib.sh` neutralises an arm
  # by no-opping this lib's functions and shadowing `uv` on PATH — but `"$REPO/.venv-arm/Scripts/python.exe"`
  # names the interpreter directly, and no PATH entry can shadow an absolute path. So `arm_check`, whose
  # whole job is to READ an arm without running it, LAUNCHED an eight-epoch GPU pretrain for every arm
  # that did this; arm_check.py's 30 s timeout then `continue`s without killing, leaving it orphaned.
  # Measured 2026-09-08: three `arm_check --only latent_register` invocations put six orphaned pretrain
  # processes on the card, and the live mask-ladder epoch went from ~400 s to over 29 minutes.
  #
  # It takes the card, which the raw calls never did — that lock alone would have refused the six.
  local tag="$1"
  shift
  hold_card
  "$REPO/.venv-arm/Scripts/python.exe" -m kneemri.training.pretrain "$@" 2>&1 \
    | tee "$MODELS/experiments/$tag.full.log"
  # THE PRETRAINER'S STATUS, NOT THE TEE'S — same rule run_train states below, and the reason it is
  # stated twice is that a pretrain that dies at the attention gate exits non-zero and a contented
  # `tee` would report the rung as survived.
  local status="${PIPESTATUS[0]}"
  release_card
  return "$status"
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
  # A CHAIN CAN BE ASKED TO STOP WITHOUT KILLING THE ARM THAT IS RUNNING.
  #
  # The card is shared with another project by arrangement, so "give it back" is a routine request and
  # it arrives mid-chain. Before this existed the only ways to honour it were to kill the driver shell
  # by pid — on a box where MSYS pids and Windows pids are different numbers for the same process, which
  # is how a live shell has already been read as dead — or to let hours of unlicensed card run out. Both
  # are worse than a file.
  #
  # Checked BEFORE the staleness re-source and before hold_card, so a yield costs neither the card nor a
  # re-parse; and checked per CALL, which is what makes it land between arms rather than inside one. The
  # arm on the card when the file appears finishes normally — that is the point, since abandoning a fit
  # ten epochs in wastes the very card time the request is trying to reclaim.
  if [ -e "$YIELD_FILE" ]; then
    echo "run_train: YIELDING before $1 - '$YIELD_FILE' exists: $(cat "$YIELD_FILE" 2>/dev/null)"
    echo "run_train: remove that file to let a chain start arms again."
    return 1
  fi
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
  # THE ARM'S OWN FLAGS, minus its name: the free-VRAM gate derives what THIS recipe reserves rather
  # than asking every arm for the same 18 GB. See `arm_needs_mib`.
  # A FITTED CHECKPOINT IS AN ASSET AND NO ARM MAY DESTROY ONE IMPLICITLY. Checked BEFORE hold_card, so
  # a refusal costs neither the card nor the queue slot behind it.
  #
  # `final_fit.sh` derives its output name from the epoch count alone — `final_ws_e${EPOCHS}.pt`, with
  # nothing about geometry in it — so re-running it at a different recipe on 2026-08-23 overwrote
  # knee-7tq's 392 px / 12-slice fit with a 336 px / 20-slice one under the same name. Nothing shipped
  # had seated the old member, so no board result lost its rebuildable path; the asset itself is simply
  # gone, and having been fitted at --fold -1 it had no offline number by which anyone would notice.
  # A name collision is silent by nature — only the file system ever knew.
  #
  # ALLOW_OVERWRITE=1 is the deliberate act: re-running an arm that failed, or replacing a fit on
  # purpose. Typing it is cheap. The point is that destruction is never what happens by default.
  local out; out="$(_out_path "${@:2}")"
  # THE LOG IS NAMED FOR THE CHECKPOINT, NOT FOR THE ARM (knee-tkyy). A smoke is the arm's own name
  # with `--studies 60 --epochs 1` added, so both wrote `<arm>.full.log` and `tee` without -a truncates:
  # the arm's first line destroyed the traceback of the smoke that was supposed to gate it. The one
  # file saying why the smoke failed was deleted by the run the smoke existed to prevent.
  #
  # `--out` is the axis that was already unambiguous — run_train REFUSES an arm whose checkpoint exists,
  # so two runs sharing a name must differ here or one of them cannot start at all.
  ARM_LOG_NAME="$(basename "${out:-$1}" .pt)"
  if [ -n "$out" ] && [ -e "$out" ] && [ -z "${ALLOW_OVERWRITE:-}" ]; then
    echo "run_train: REFUSED $1 - '$out' already holds a checkpoint. Set ALLOW_OVERWRITE=1 to replace it, or give this arm a distinct --out."
    return 1
  fi
  # The arm's NAME, which `hold_card "${@:2}"` drops on purpose — everything downstream of it wants the
  # trainer's flags, not the name. The claim file is the one exception: it is written for a PEER to read,
  # and a claim reading `job: --fold` names nothing anyone could act on. Measured 2026-08-24T12:10Z.
  CLAIM_JOB="$1"
  hold_card "${@:2}"
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
    | tee "$(_arm_log "$name")" \
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
    # The STATUS is not the cause and on this platform it is routinely not even a hint: a native abort
    # inside the CUDA driver leaves the shell holding 127, which is `command not found` and which the
    # environment note at the top of this file spent three arms teaching everyone to read as a rewritten
    # venv. On 2026-09-03 `ortho_pool_read` died of `RuntimeError: CUDA driver error: out of memory` and
    # was diagnosed twice from the marker alone - as a gates kill, then as a host spill - while the real
    # line sat 295 lines up the same file. `ortho_fresh_lowlr` is very likely the same misreading, filed
    # as a method failure. The exception line is already IN the log; nothing was carrying it to the one
    # line an operator reads, so it is carried here.
    # Two shapes, because the second one is how `ortho_fresh_lowlr` spent a week filed as a method
    # failure. It raised no exception at all: it asked for 31.9 GB on a 32.6 GB card, waited the full
    # 1800 s for VRAM that was never coming, and gave up having trained zero epochs. The last line of
    # its own log said exactly that. So fall back to it when there is no exception to quote.
    local cause
    cause="$(grep -E "^[A-Za-z_.]*(Error|Exception):" "$(_arm_log "$name")" | tail -1)"
    [ -n "$cause" ] || cause="$(grep -vE "^(LIB |ARM )" "$(_arm_log "$name")" | tail -1)"
    # The hint must name the file the two greps above just READ, not `$name`. `_arm_log` spells the log
    # for the CHECKPOINT while `$name` is the run_train name (knee-tkyy), so the two differ whenever
    # `--out` carries a seat the arm's name does not: grand_recipe died pointing readers at
    # `grand_recipe.full.log` while its log was `grand_recipe_s2026.full.log`, and a path that does not
    # exist reads as a lost log rather than a failed arm.
    local verdict="ARM FAILED $name (trainer exit $status)${cause:+ - $cause} - see $(_arm_log "$name")"
    _mark "$name" "$verdict"
    # An arm death is the one event with a deadline attached: the card is free NOW and the queue behind
    # it is spending wall-clock on nothing. grand_recipe died eleven steps into a reserved six-hour
    # window and the line sat unread until the window was over.
    _post "FAILED-$name" "$verdict"
    return "$status"
  fi
  # A trainer can also exit 0 having produced nothing useful. The best-score line is the artefact the
  # ledger quotes, so its absence is a failure whatever the exit code says.
  if ! grep -q "best holdout" "$(_arm_log "$name")"; then
    local verdict="ARM FAILED $name (exit 0 but no 'best holdout' line) - see $(_arm_log "$name")"
    _mark "$name" "$verdict"
    _post "FAILED-$name" "$verdict"
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
  best="$(grep -m1 'best holdout' "$(_arm_log "$name")")"
  checkpoint="${best##*-> }"
  checkpoint="${checkpoint%% |*}"
  if [ ! -f "$checkpoint" ]; then
    _mark "$name" "SEAT VALUE skipped: no checkpoint at '$checkpoint'"
    _post "DONE-$name" "ARM DONE $name - UNPRICED: no checkpoint at '$checkpoint'"
    return 0
  fi
  # --no-sync for the same reason run_train uses it: another chain may already be training, and a
  # re-sync of .venv under its interpreter kills it mid-epoch.
  local report
  if ! report="$(uv run --no-sync --project "$REPO" python -m kneemri.kaggle.seat_value \
      --candidates "$checkpoint" 2>&1)"; then
    report="SEAT VALUE unavailable for $name: $report"
  fi
  _mark "$name" "$report"
  # THE CUT IS TAKEN HERE, NOT REMEMBERED. `seat_value` answers "does this earn a seat"; it does not
  # print the score the ledger quotes, and a number in this repo is MIDPOINT unless it says otherwise
  # (knee-f9aao) - so a finished arm used to leave no `ledger75` reading at all, and taking one was a
  # second command nobody ran. Nineteen arms finished that way on 2026-09-15.
  local stem priced
  stem="$(basename "$checkpoint" .pt)"
  if ! priced="$(uv run --no-sync --project "$REPO" python -m kneemri.evaluation.ledger75 \
      --holdouts "$MODELS" --only "$stem" 2>&1)"; then
    priced="LEDGER75 unavailable for $stem: $priced"
  fi
  _mark "$name" "$priced"
  # Both readings, one letter, one interrupt. The point of posting the DONE rather than only the FAILED
  # is that an arm that SUCCEEDS is the one whose result has to be read before the next arm is chosen -
  # and success is exactly the case that looks like it needs no attention.
  _post "DONE-$name" "ARM DONE $name -> $checkpoint

$report

$priced"
}

_arm_log() {
  # The file a run writes, named for the CHECKPOINT it produced rather than for the arm that produced
  # it (knee-tkyy). Falls back to the arm name for a run with no `--out`, which is the only case where
  # the two were never distinguishable anyway.
  echo "$LOGS/${ARM_LOG_NAME:-$1}.full.log"
}

_mark() {
  # Write an arm's verdict to BOTH the terminal and the arm's own log.
  #
  # It used to go only to stdout, which lands wherever the caller redirected - so an arm's full.log
  # held every epoch and no verdict, and a gate pointed at that log waited forever for a marker that
  # was never going to arrive there. Found while queueing the resolution arm: the GPU would simply
  # have gone idle with nothing reporting an error.
  local name="$1" message="$2"
  echo "$message" | tee -a "$(_arm_log "$name")"
}

_post() {
  # Deliver an arm's verdict to `inbox/main/`, which is the only channel in this repo that INTERRUPTS.
  #
  # `_mark` and `_seat_value` were both already correct and both already useless on the night that
  # mattered: on 2026-09-15 twenty arms wrote DONE markers and priced seats into their own logs, and
  # nineteen finished checkpoints then sat unread for six hours, `labelwise_dense_drop40_long36` among
  # them - first priced only because the owner asked. A log is a PULL channel. Every one of this repo's
  # sixteen checks is a pull channel. They all work; the operator is what fails, and writing a
  # seventeenth thing for the same operator to remember to read is the same defect with more prose.
  #
  # A letter is the one PUSH: the box's listener prints the body and exits, which the harness delivers
  # as a task notification that arrives mid-turn whether or not anyone thought to look. So the verdict
  # a human must react to goes here as well as to the log, and reacting no longer requires remembering.
  #
  # The letter SURVIVES an unarmed listener - it sits in the box and delivers on the next arm - which
  # is strictly better than a log line that must be sought by name. `--box main` is the execution
  # lane's own box (`inbox/agents.csv`); an arm writes into it and never watches it.
  #
  # The stamp is READ, never estimated: an estimated stamp drifts with session length and breaks the
  # cross-side sort (measured +4 h after three hours, D338). Atomic `.part` then `mv`, so the listener
  # can never print half a letter it caught mid-write.
  local subject="$1" body="$2"
  local box="$REPO/inbox/main" stamp
  [ -d "$box" ] || return 0
  stamp="$(date -u +%Y%m%dT%H%MZ)"
  local letter="$box/$stamp-arm-$subject.txt"
  printf '%s\n' "$body" > "$letter.part" && mv "$letter.part" "$letter"
}

wait_for() {
  # wait_for <completion-marker> <file>. Blocks until the marker appears, OR until the paired failure
  # marker does. Returns 0 on the completion marker, 3 on the failure marker.
  #
  # IT USED TO WAIT ONLY FOR SUCCESS, AND A CRASHED ARM THEN HELD THE CARD IDLE FOREVER. `run_train`
  # writes `ARM DONE <name>` on success and `ARM FAILED <name> (...)` on failure (:631, :637, :642) into
  # the same file, so a gate on the DONE spelling never releases when the arm dies - it just sleeps 60 s
  # in a loop with nothing running behind it and `queue_check` reading OK, because a live process gating
  # an owed arm is all that check can see. Measured 2026-09-08 (researcher D223 G.1): `after_trunkdraws`
  # and `after_instruments` were both blocked ~15 h on `ARM DONE deep32_batch6`, whose arm had died at
  # 02:02:56Z on a FileNotFoundError. It cost no card only because a third watcher on a different gate
  # happened to own the arm they were both waiting to run.
  #
  # A BROKEN GATE RELEASES THE QUEUE RATHER THAN ABANDONING IT. The gate exists to serialise the CARD,
  # not to express a data dependency: the producer failing means the card is free NOW. An arm that truly
  # consumes the producer's output dies at its own `require` or in `after.sh --smoke`, which is a loud,
  # bounded failure; an idle card is a silent, unbounded one.
  local marker="$1" file="$2"
  local failed="${marker/ARM DONE /ARM FAILED }"
  while true; do
    grep -q "$marker" "$file" 2>/dev/null && return 0
    if [ "$failed" != "$marker" ] && grep -q "$failed" "$file" 2>/dev/null; then
      echo "GATE BROKEN: '$failed' in $file - the arm this gate waits on FAILED. Releasing the queue"
      echo "GATE BROKEN: because the card is free; a dependent arm will die at its own require/smoke."
      grep -m1 "$failed" "$file"
      return 3
    fi
    sleep 60
  done
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
  key="$(uv run --no-sync python -m core.data.corpus.build_cache --print-key   --image-size "$size" --slices "$slices" --scheme "$scheme" --crop-mm "$crop" --centre "$centre")"
  [ -n "$key" ] || { echo "wait_for_cache: the builder named no key"; return 1; }
  until [ -f "$PROCESSED/mask_${key}.npy" ]; do sleep 60; done
}
