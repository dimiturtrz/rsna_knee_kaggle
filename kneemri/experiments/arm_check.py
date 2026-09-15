"""Prove every committed arm's flags before a queue costs GPU: `python -m kneemri.experiments.arm_check`.

This project's most expensive failures are arms that fail SILENTLY or late. Arm U printed `ARM U DONE`
after dying at epoch 7 of 24. A `str.replace` that matched nothing left `dinov3.sh` writing one name
for every read size, so a second probe began overwriting the run it was being compared against. A
package snapshot without a pyproject broke `uv run` and killed two queued arms in under a second. An
arm invoked with no arguments iterated zero times and exited 0 with an empty log.

A flag typo belongs to that family: it costs a queue slot to discover, and it discovers it after the
card has been held. So each arm is sourced against `experiments/stub_lib.sh`, whose `run_train` prints
its arguments instead of training, and the resulting line is handed to the REAL `TrainingOptions.parse`.

Sourcing rather than parsing the file, because the shell is what runs it: an arm that dies on a quoting
error is one this project has already lost two hours to, and only bash finds that.

The accepted flags are then asked a second question: is the cache that reading names actually decoded?
`main` takes the card BEFORE it maps the cache, so an undecoded reading costs the slot in the same way
a typo does — and it is a `FileNotFoundError` on a memory map rather than anything an arm anticipates.
"""

import contextlib
import io
import logging
import os
import shlex
import sys
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from core import cli
from core.data.corpus.study_cache import CorpusSplit, StudyCache
from core.data.labels.ruler import DEFAULT_LABELS, Ruler
from core.denominator import Denominator
from core.paths import DataRoot
from kneemri.experiments.arm_diff import ArmDiff
from kneemri.experiments.arm_header import ARMS, ArmHeader
from kneemri.experiments.arm_shadow import ArmShadow
from kneemri.experiments.arm_smoke import SMOKE_STUDIES, ArmSmoke
from kneemri.experiments.arm_verdict import ArmVerdict
from kneemri.experiments.switch_coverage import SwitchCoverage
from kneemri.spawn import Spawn
from kneemri.training.device_budget import GB_PER_SLICE
from kneemri.training.options import TrainingOptions
from kneemri.training.run_setup import RunSetup

logger = logging.getLogger(__name__)

PRICE_MARKS = ("PRICE:", "P(clears")
"""What a priced bead says: a cost AND the odds it clears the bar. Both, because a cost alone is the
argument that let the 140 mm build run unpriced. knee-yuqf carries the format."""

STUB = Path("experiments/stub_lib.sh")
SHELLS = Path("experiments")
"""Every shell file an arm's run touches, arms and the libs they source alike — the CR check's subject.

Named apart from `ARMS` because the file that has actually cost a decode is `lib.sh`, which no arm is
but every arm sources.
"""
PATHS = Path("paths.yaml")
TIMEOUT_SECONDS = 30
"""An arm that has not printed its invocation by now is doing something a lint must not wait for."""

SPAWN_WORKERS = 4
"""How many arm declarations are read at once — spawn latency rather than compute, kept LOW on
purpose. The card is untouched: an arm sourced against the stub prints its
flags and exits without importing torch. Serially the 300-odd arms plus the handful that sit out the
full `TIMEOUT_SECONDS` put this lint past five minutes, which is long enough to be read as a hang and
skipped before arming — and a pre-queue check nobody runs is the queue slot it was built to save.
FOUR, not sixteen: at sixteen this fan-out saturated the box badly enough that `nvidia-smi` stopped
answering within ten seconds, and the live arm's own VRAM heartbeat raised on that timeout and died
at epoch 7 of 12. The heartbeat is now tolerant (`card_memory.free_mib`), but a lint that can stall
the machine it lints is wrong at both ends, and four is already most of the speed-up."""

SMOKE_DECODE_WORKERS = 8
"""Fewer than a corpus build takes, because a smoke decode may share the machine with a live arm."""

OPERATING_POINT_FLAGS = ("--slices", "--band-low", "--band-high")
"""The flags whose PARSER DEFAULTS are the pre-settlement operating point, so silence about them is a
recipe rather than an omission.

`TrainingOptions` defaults to 12 slices and `SLICE_BAND` 0.20-0.80. That is not a neutral starting
point — it is the operating point every arm was read at BEFORE the span and density work settled on 20
slices and 0.10-0.90, and knee-aixw is the bead recording that a lever's sign is a property of the
operating point it was measured at. calib_span is the proof: `calibrated_targets` measured +0.0038 at
the old point, earned a seat in a champion package on that number, and re-measured -0.0020 at the one
we ship with its per-label clause inverted.

So an arm that names neither takes the old point in SILENCE, and the run's log says nothing a reader
could catch it by. knee-5895 is that defect on a live script: `final_fit.sh` states no slice count and
no band, so it would fit the final model at 12 slices in 0.20-0.80 — and at `--fold -1` there is no
score, so nothing downstream would have caught it either."""

GIT_BASH = Path("C:/Program Files/Git/usr/bin/bash.exe")
"""The interpreter the arms are written for, named explicitly.

A bare `bash` resolves to WSL's on this machine, and WSL here is wedged often enough to have its own
note — so every arm timed out and this lint reported "0 arms checked, 0 refused". A clean pass over
nothing is the exact failure it was built to catch, produced by the tool itself on its first run.

The SECOND time it produced that same pass was subtler: run from PowerShell rather than Git Bash, the
inherited PATH carries no `/usr/bin`, so the stub lib's `dirname` and `mktemp` are not found and every
arm prints nothing. Hence `bin` below — the interpreter's own directory goes on the PATH it is handed,
so the shell is the same shell whatever launched python."""


@dataclass(frozen=True)
class ArmReading:
    """Where one arm's pixels come from: the root it was pointed at, and the cache its flags name.

    The pair rather than the cache alone, because the smoke needs both halves — the cache to ask
    whether it is on disk, and the root to decode it out of raw when it is not. Resolved once per arm,
    since asking an arm what it reads means driving the real parser over its real flag line.
    """

    root: DataRoot
    cache: StudyCache

    @property
    def key(self) -> str:
        """The cache filename this reading resolves to, which is what an UNCACHED report names."""
        return self.cache.key

    def decoded(self) -> bool:
        """Whether the cache this arm would read is already on disk."""
        return self.cache.decoded()

    def truncated(self, first: int) -> "ArmReading":
        """The same reading over the split's first N studies — a different file, by name."""
        return ArmReading(self.root, replace(self.cache, first=first))

    def decode(self, workers: int = SMOKE_DECODE_WORKERS) -> None:
        """Build this cache out of raw. Under a minute for a smoke's sixty studies, two hours for all.

        `workers` is an argument rather than the constant because a process pool is the expensive part
        of a decode, not the pixels: a test that wanted two studies spawned eight interpreters, each
        re-importing torch on Windows' spawn, and one unit run cost seven and a half hours.
        """
        split = CorpusSplit(self.root.raw())
        self.cache.build(split.source, split.identifiers(), split.planes(), workers=workers)


class ArmCheck:
    """Every committed arm's train invocation, and whether the trainer would accept it."""

    def __init__(self, arms: Path = ARMS, stub: Path = STUB, shells: Path = SHELLS) -> None:
        self.arms = arms
        self.stub = stub
        self.shells = shells
        self.headers = ArmHeader(arms)
        self.diff = ArmDiff(self.headers)

    def _environment(self) -> dict[str, str]:
        """The inherited environment, plus ARM_LIB, the resolved data root, and coreutils on PATH.

        The stub lib fabricates a processed root when it is handed none, which is enough to prove flags
        parse and is what CI gets. `--smoke` RUNS those flags, so it needs the real one — resolved here
        rather than by the stub, because the stub shadows `uv` and every arm would pay a subprocess.
        """
        path = os.environ.get("PATH", "")
        if GIT_BASH.exists():
            path = f"{GIT_BASH.parent}{os.pathsep}{path}"
        environment = os.environ | {"ARM_LIB": str(self.stub.resolve()), "PATH": path}
        if PATHS.is_file():
            roots = DataRoot.from_config(PATHS)
            environment["ARM_CHECK_PROCESSED"] = roots.processed().as_posix()
            environment["ARM_CHECK_MODELS"] = roots.models().as_posix()
        return environment

    def queued(self, entries: list[str]) -> dict[str, str]:
        """The flag lines of arms invoked AS A QUEUE WOULD invoke them — with their own arguments.

        THE HALF THIS CHECK COULD NOT SEE. `after.sh` takes an arm as one argv element carrying its
        arguments (`"experiments/arms/resolution.sh 448"`), and every check here runs the arm files
        BARE. So an arm's non-default rungs — 448 px, a different LoRA rank, a longer schedule — were
        queued on flags that nothing had ever parsed, which is exactly the flag-typo-costs-a-queue-slot
        failure this module exists to prevent, at the one moment the arm is not the shipping one.

        Takes the entry in `after.sh`'s spelling rather than a name plus a list, so what is checked is
        the string that will be queued, character for character.

        AN ENTRY THAT IS NOT A SCRIPT IS AN ERROR HERE, where it can be attributed. Passing an arm's
        NAME (`band_dense_long`) instead of its path is the natural mistake — every other flag on this
        tool speaks names — and bash then runs nothing, prints nothing, and leaves the invocation set
        empty. The failure surfaced downstream in `selected` as "no arm printed an invocation named
        band_dense_long", which blames `--only` for a typo in `--as-queued` and reads as a missing arm
        rather than a wrong spelling. Same rule as `selected`: a name matching nothing is an error, and
        it belongs at the argument that was actually wrong.
        """
        commands = [shlex.split(entry, posix=False) for entry in entries]
        unscripted = [entry for entry, command in zip(entries, commands, strict=True) if not Path(command[0]).is_file()]
        if unscripted:
            message = (
                f"--as-queued takes a queue entry, a SCRIPT PATH carrying its arguments "
                f"('experiments/arms/resolution.sh 448'), and these name no file: {', '.join(unscripted)}"
            )
            raise ValueError(message)
        return self._printed(commands)

    def invocations(self) -> dict[str, str]:
        """Each arm's `run_train` flag line, by arm name. Arms that never call it are absent.

        Some arms drive `uv run` directly rather than through `run_train` — the diversity curve, the
        cache build — and have no flags to check. They time out or print nothing, and are skipped
        rather than reported: a lint that cries about files it cannot check gets ignored.

        THIS METHOD WAS THE LAUNCHER, and the paragraph that stood here saying otherwise was wrong —
        knee-q4c3u was filed as a P0, closed as no defect, and the defect was real. `stub_lib.sh` puts a
        no-op `uv` ahead of the real one, so every arm reaching its interpreter through `uv` is inert.
        `after_ortho_encode.sh` named `.venv/Scripts/python.exe` directly, and the exoneration was that
        `set -u` against an unbound `${1}` aborts it before that line. IT DOES NOT. The expansion sits
        inside a pipeline, so `set -u` kills the SUBSHELL and not the script; `tasklist` fails, the
        leading `!` turns that failure into true, and the `until` loop falls through on its first
        iteration into a six-hour `build_ortho_embeddings`. Reproduced 2026-09-04 by running the file
        bare, which is exactly how this method runs it: the encode started within six seconds.

        The fix is in the shell — an explicit `[ $# -ge 1 ]` guard, and the file moved OUT of
        `experiments/arms/` to `experiments/`, since a watcher is not an arm and has no flags to read.
        Both halves matter: the glob is what made a lint that reads declarations execute a card job.
        Do not "fix" a recurrence by moving the cwd — arms `cd "$REPO"` as their second line, which
        defeats that and nothing else. Check instead that nothing under `arms/` reaches an interpreter
        without going through `uv`.
        """
        return self._printed([[str(arm)] for arm in sorted(self.arms.glob("*.sh"))])

    def _spawned(self, command: list[str]) -> str:
        """One arm's stdout, or empty when it hung past `TIMEOUT_SECONDS`."""
        done = Spawn().bounded(
            [str(GIT_BASH) if GIT_BASH.exists() else "bash", *command],
            TIMEOUT_SECONDS,
            # The INHERITED environment plus ARM_LIB. Replacing it outright leaves Git Bash
            # without the PATH it needs to find its own coreutils, and every arm then produces
            # nothing - which this lint reports as "0 arms checked", a clean pass over nothing.
            env=self._environment(),
        )
        return "" if done is None else done.stdout

    def _printed(self, commands: list[list[str]]) -> dict[str, str]:
        """Run each arm command against the stub lib and collect the `name<TAB>flags` lines it prints.

        CONCURRENTLY, because the wall-clock is the whole usability question. Every invocation is an
        independent Git Bash reading declarations against the stub, so the work is spawn latency rather
        than compute; serially, 300-odd arms plus the few that sit out the full `TIMEOUT_SECONDS` put
        this past five minutes, which is how a pre-queue lint came to be read as a HANG and skipped
        before arming (knee, 2026-09-09 — a faulthandler dump showed it inside `subprocess.communicate`
        on arm N of 305, not stuck). The results are collected into an ordered dict AFTER the pool
        drains, in the commands' own order, so the duplicate-flag verdict sees the same sequence it saw
        when this ran one at a time.
        """
        # BEFORE THE POOL, because the pool is what RUNS them: checking such an arm means running it,
        # and running it is the incident `arm_shadow` exists to prevent. `reported` names them in `main`.
        unsafe = {str(arm) for arm in ArmShadow(self.arms).unshadowed()}
        commands = [command for command in commands if command[0] not in unsafe]
        found: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=SPAWN_WORKERS) as pool:
            outputs = list(pool.map(self._spawned, commands))
        for command, output in zip(commands, outputs, strict=True):
            for line in output.splitlines():
                name, tab, flags = line.partition("\t")
                # The stub SHELL-QUOTES what it prints, so a flag value containing a space keeps its
                # argument boundary; the "is this an invocation" test therefore has to look at the
                # first TOKEN rather than at the raw line, which now begins with a quote character.
                first = shlex.split(flags)[:1]
                if tab and first and first[0].startswith("--"):
                    found[name] = flags
                    self.headers.record(name, Path(command[0]))
        return found

    @staticmethod
    def selected(invocations: dict[str, str], names: list[str] | None) -> dict[str, str]:
        """The named arms alone, or all of them when nothing was named.

        A NAME THAT MATCHES NOTHING IS AN ERROR, not an empty selection: the whole point of `--only` is
        to smoke ONE arm when the card is scarce, and a typo that silently smokes zero arms exits 0
        with "0 arms checked" — the clean-pass-over-nothing failure this module already carries two
        scars from. Filtered on the name the arm PRINTS rather than its filename, because that is the
        name the ledger, the checkpoint and the rest of this check all speak.
        """
        if names is None:
            return invocations
        missing = [name for name in names if name not in invocations]
        if missing:
            message = f"no arm printed an invocation named {', '.join(missing)}"
            raise ValueError(message)
        return {name: flags for name, flags in invocations.items() if name in names}

    def _parsed(self, flags: str) -> tuple[TrainingOptions | None, str]:
        """The options an arm's flags produce, or the parser's own complaint about them.

        One place that drives the real parser, because both halves of the check need it: whether the
        trainer accepts the arm, and what reading it accepted.
        """
        argv = sys.argv
        sys.argv = ["train", *shlex.split(flags)]
        # Bound BEFORE the try, because the parser exits by raising: bound inside the `with`, the
        # handler reads a name the checker cannot prove was ever assigned.
        complaint = io.StringIO()
        try:
            with contextlib.redirect_stderr(complaint):
                return TrainingOptions.parse(), ""
        except SystemExit as exit_code:
            if not exit_code.code:
                return None, ""
            lines = complaint.getvalue().strip().splitlines()
            return None, lines[-1] if lines else "rejected without a message"
        finally:
            sys.argv = argv

    def refused(self, invocations: dict[str, str]) -> dict[str, str]:
        """The arms whose flags `kneemri.training.train` would reject, with the parser's own complaint.

        A pure function of the invocations, so it is testable without a shell — which matters, because
        the unit gate refuses tests that shell out and this is the half that holds the logic.
        """
        complaints = ((name, self._parsed(flags)[1]) for name, flags in invocations.items())
        return {name: complaint for name, complaint in complaints if complaint}

    def carriage_returns(self) -> dict[str, int]:
        """The shell files carrying a CR on disk, by path, with the count — the defect git cannot show.

        A CR inside a brace group or a subshell is a bash syntax error, and `.gitattributes` normalises
        line endings ON THE WAY TO THE INDEX. So a CRLF working copy produces no diff, no status entry
        and no warning at review time, while the shell reads the file as it is on disk and dies. Every
        layer anyone checks agrees the tree is clean.

        It surfaces where it costs most. On 2026-08-19 `build_cache.sh` decoded all 4407 studies and
        THEN died on its own epilogue — so `ARM DONE build_cache_<key>` was never echoed, the
        mask-written post-condition never ran, and anything gated on that marker would have waited
        forever behind a cache already on disk (knee-rlg5). Twelve of the tree's shell files were in
        that state, `experiments/lib.sh` among them.

        The whole file is read rather than the parsed invocation, because the constructs a CR breaks
        are the ERROR HANDLING and the completion markers — the lines that only run when something
        notable happened, and so the lines a smoke never reaches.
        """
        counted = ((path, path.read_bytes().count(b"\r")) for path in sorted(self.shells.glob("**/*.sh")))
        return {str(path.as_posix()): count for path, count in counted if count}

    @staticmethod
    def unpriced(descriptions: Mapping[str, str]) -> list[str]:
        """The beads whose text carries no wall-clock/gain price, given each bead's own description.

        PURE, and that is why the `bd` call lives in `main()`: the unit gate refuses a test that shells
        out, and this is the half holding the judgement. It takes the text rather than the id.

        Two markers, not one. `PRICE:` alone would be satisfied by any bead that happens to use the word,
        and the number that actually governs the decision is the odds — a bead saying "~1 h of card" with
        no `P(clears 1.5%)` has stated a cost and no expected gain, which is exactly the argument I made
        for the 140 mm build and exactly the one the rule exists to refuse (knee-qxse). knee-yuqf is the
        format both markers were read off.
        """
        return sorted(bead for bead, text in descriptions.items() if not all(mark in text for mark in PRICE_MARKS))

    @staticmethod
    def _descriptions(beads: set[str]) -> dict[str, str]:
        """Each bead's own text, read out of `bd`. The one place this tool talks to the tracker.

        Kept apart from `unpriced` so that half stays pure and unit-testable — the gate refuses a unit
        test that shells out. A bead `bd` cannot show comes back as empty text, which reads as unpriced:
        an arm naming an id that does not resolve is exactly as unaccountable as one naming none.
        """

        def shown(bead: str) -> str:
            # fixed argv, a bead id matched by ArmHeader's BEAD_LINE, no shell
            done = Spawn().ran(
                ["bd", "show", bead],
                capture_output=True,
                text=True,
                check=False,
            )
            return done.stdout

        return {bead: shown(bead) for bead in beads}

    def prices(self, invocations: Mapping[str, str]) -> list[ArmVerdict]:
        """The arms naming no bead, and those naming an unpriced one — the two halves of one rule.

        The ONE place this tool talks to the tracker, joined to the pure detectors either side of it:
        `beads` reads the headers, `unpriced` judges the text `bd` returns, and this is the read between
        them. Both halves come back as verdicts rather than as a refusal, because the rule exists to stop
        MOMENTUM at the moment of arming and nothing here knows whether this run is arming (knee-qxse).
        """
        beads = self.headers.beads(invocations)
        named = {name: bead for name, bead in beads.items() if bead}
        unpriced = set(self.unpriced(self._descriptions(set(named.values()))))
        return [
            ArmVerdict(
                "UNBEADED",
                {
                    name: "names no `# bead: knee-xxxx`, so nothing records what it expects"
                    for name, bead in beads.items()
                    if bead is None
                },
                clause="name no bead at all",
                arming_refuses=True,
            ),
            ArmVerdict(
                "UNPRICED",
                {
                    name: f"{bead} states no PRICE / P(clears ...) — a cost, no gain"
                    for name, bead in named.items()
                    if bead in unpriced
                },
                clause="name a bead stating a cost with no expected gain",
                bead="knee-qxse",
                arming_refuses=True,
            ),
        ]

    def verdicts(
        self,
        invocations: dict[str, str],
        against: Mapping[str, str],
        refused: Mapping[str, str],
        uncached: Mapping[str, str],
        *,
        require_caches: bool,
    ) -> list[ArmVerdict]:
        """Every check this tool runs, phrased — the sentences an author reads, beside the rules.

        Apart from `main` because that entrypoint's job is argv and an exit code, and it had grown to
        hold the phrasing of seven checks as well: at cyclomatic 21 it was the most tangled function in
        the tree, and its complexity was the interleaving rather than any one rule. `refused` and
        `uncached` are handed in rather than recomputed here, because the smoke and the closing tally
        both want the raw dicts and parsing a hundred arms twice to re-derive them is a hundred parses.
        """
        overstated = self.overstated(invocations)
        worst = max(overstated, key=lambda name: overstated[name], default="")
        return [
            ArmVerdict("REFUSED", refused, always=True, refuses=True),
            ArmVerdict(
                "UNCACHED",
                {name: f"nothing decoded at {key}" for name, key in uncached.items()},
                always=True,
                refuses=require_caches,
            ),
            ArmVerdict(
                "CRLF",
                {
                    path: f"{count} carriage returns on disk — bash reads the file as it is"
                    for path, count in self.carriage_returns().items()
                },
                bead="knee-rlg5",
                always=True,
                refuses=True,
            ),
            ArmVerdict(
                "OVERSTATED",
                {name: f"asks {over:.2f} GB above what its own --slices reserve" for name, over in overstated.items()},
                clause=f"name a --vram-gb above what their own --slices reserve, worst {worst} by "
                f"{overstated.get(worst, 0.0):.2f} GB — the flag predates DeviceBudget.needed() and no "
                f"arm needs it, so drop it",
                bead="knee-ylp0",
            ),
            ArmVerdict(
                "UNSTATED",
                {
                    name: f"says nothing about {', '.join(missing)}, so it inherits the pre-settlement "
                    f"12 slices / 0.20-0.80 in silence"
                    for name, missing in self.unstated(invocations).items()
                },
                clause=f"name none of {', '.join(OPERATING_POINT_FLAGS)}, so they inherit the "
                f"pre-settlement 12 slices / 0.20-0.80 in silence",
                bead="knee-aixw",
            ),
            ArmVerdict(
                "UNMET-GATE",
                self.headers.gated(invocations),
                clause="declare a `# gate:` the board does not meet",
                bead="knee-1e3m",
                always=True,
                arming_refuses=True,
            ),
            ArmVerdict(
                "MISMATCHED",
                {
                    name: f"undeclared against its own control — {'; '.join(found)}"
                    for name, found in self.diff.mismatched(invocations, against).items()
                },
                clause="differ from the control they name without a `# deliberate:` line saying so",
                bead="knee-f8j9",
                arming_refuses=True,
            ),
            ArmVerdict(
                "MOVING-RULER",
                self.moving_ruler(invocations),
                clause="re-map their labels without pinning --eval-labels, so their holdout truth moves "
                "with their targets and their score is two rulers rather than a result",
                bead="knee-df7t",
                always=True,
                refuses=True,
            ),
            ArmVerdict(
                "TWIN",
                self.diff.twinned(invocations, against),
                clause="carry a flag line another arm already carries, differing only in the file they "
                "write — so the card is spent twice on one recipe under two names",
                bead="knee-7uoq",
                always=True,
                refuses=True,
            ),
            *self.prices(invocations),
        ]

    def readings(self, invocations: dict[str, str]) -> dict[str, ArmReading]:
        """Where each arm's pixels would come from, by arm name — its data root and the cache it names.

        Silent about arms the parser refuses, and about any checkout where the data root cannot be
        resolved, which is CI and every clone without a `paths.yaml` — a lint that fails on the absence
        of gitignored data gets switched off.

        One place that resolves a reading, because both consumers need the same one: the UNCACHED
        report, and the smoke, which decodes a truncation of it when nothing is there.
        """
        readings: dict[str, ArmReading] = {}
        for name, flags in invocations.items():
            options = self._parsed(flags)[0]
            if options is None or not options.paths.is_file():
                continue
            root = DataRoot.from_config(options.paths)
            readings[name] = ArmReading(root, options.cache(root.processed()))
        return readings

    def uncached(self, invocations: dict[str, str]) -> dict[str, str]:
        """The arms whose reading names a cache that is not on disk, by arm name and cache key.

        THE OTHER LATE FAILURE, and the same price as a flag typo. `--image-size`, `--slices`,
        `--crop-mm`, `--scheme` and `--tie` together pick a cache file, and `main` resolves the CARD
        before it maps that file — so a reading nothing has been decoded for holds the GPU, raises on
        the memory map, and is discovered as a burned queue slot. Multiscale makes it live: the twelve-
        slot scheme is implemented and no twelve-slot cache has ever been built.
        """
        return {name: reading.key for name, reading in self.readings(invocations).items() if not reading.decoded()}

    def overstated(self, invocations: dict[str, str]) -> dict[str, float]:
        """The arms asking for more VRAM than their own slice count needs, by how many GB.

        A HAND-TYPED CAP IS A THIRD STATEMENT OF A FACT `--slices` ALREADY MAKES, and the third one goes
        stale: `depth_regions_narrow` is `depth_regions` with the slices dropped to 12 and the 20-slice
        cap of 28 copied along, 10 GB above what a 12-slice step reserves. That was free on an empty
        card — `DeviceBudget.await_headroom` waits for `free >= vram_gb` and an overstatement just
        passes — and stopped being free when `biohub_kaggle` took ~13 GB of the 32: the arm now polls
        for memory it never needed, burns the wait, and starts into the PCIe-paging crawl the cap
        exists to prevent. `depth_regions_span` was queued at 28 on a 12-slice recipe and caught by
        hand; this is that catch, mechanised.

        Measured against `DeviceBudget.needed`, which already carries one slice of slack. Only
        OVERSTATEMENT is reported: asking under the derived need is what `band_dense` does at 28
        against 28.26, and it peaked at 26.92.

        IT REPORTS 38 OF 50 ARMS, AND THAT IS THE TRUE ANSWER RATHER THAN A LOOSE BAR. I first read the
        count as noise and tried to legislate a threshold under which slack was "inside the 20 / 26 /
        28 convention" — but that convention is exactly the defect: every one of those numbers was
        picked as a fraction of the CARD, months before `DeviceBudget.needed` derived a cap from the
        RECIPE, so 26 on a 12-slice arm is 8.4 GB over need for the same reason 28 is 10.4 — not
        because one was copied and the other rounded. There is no honest line between them. So this
        returns all of them and `main` says it once, in a sentence naming the worst, instead of
        thirty-eight times: the flag is obsolete tree-wide, and the fix is to delete it tree-wide
        (knee-ylp0), not to tune a bar until the report is quiet.
        """
        asked: dict[str, float] = {}
        for name, flags in invocations.items():
            options = self._parsed(flags)[0]
            if options is None or not options.vram_gb:
                continue
            needed = RunSetup(options).budget().needed()
            if options.vram_gb > needed + GB_PER_SLICE:
                asked[name] = round(options.vram_gb - needed, 2)
        return asked

    @staticmethod
    def unstated(invocations: dict[str, str]) -> dict[str, list[str]]:
        """The arms that never say what operating point they run at, by which flags they leave silent.

        IT ASKS THE FLAG LINE AND ITS RECIPE FILE, NOT THE PARSED OPTIONS, and that is the whole check.
        After parsing, an arm that asked for 12 slices and an arm that asked for nothing are the same object — which is
        exactly why this defect survives every gate the project already runs. Only the text distinguishes
        a stated operating point from an inherited one.

        NOT A VALUE CHECK. An arm may legitimately run at 12 slices or a narrow band; `band_span` does,
        on purpose, and reading it as wrong would make this a rule against experiments. What it may not
        do is arrive there by silence, because a lever measured at a point nobody wrote down cannot be
        read against anything later — see `OPERATING_POINT_FLAGS` for the case that cost a champion seat.
        """
        silent = {
            name: [flag for flag in OPERATING_POINT_FLAGS if flag not in TrainingOptions.spelled(shlex.split(flags))]
            for name, flags in invocations.items()
        }
        return {name: missing for name, missing in silent.items() if missing}

    @classmethod
    def moving_ruler(cls, invocations: dict[str, str]) -> dict[str, str]:
        """The arms that re-map their training targets without pinning what their holdout is scored by.

        `Ruler` holds what makes two tables different rulers and counts the cells; this is the arm-side
        half, and it asks the FLAG LINE the way `unstated` does — after parsing, an arm that pinned
        `--eval-labels` to the default table and one that inherited it are the same object.

        WHAT IT COST. `recal_holdannot` (knee-8wa0) ended `best holdout mean AUC 0.8964` against a
        control's 0.8487 — a +0.0477 that would have entered the ledger as the largest single-arm gain
        the project has recorded. It is two rulers. Re-scored against the shared table the gain is
        +0.0009 on the proxy and +0.0105 on gold. `run_comparison` refuses the pair outright, which is
        the only reason it was caught, and it refuses AFTER the card has been spent. `blendlabels` got
        this right on 2026-08-25 by pinning `--eval-labels`; nothing made that a rule until now.
        """
        moved: dict[str, str] = {}
        for name, flags in invocations.items():
            values = ArmDiff.values(flags)
            table = Path(values.get("--labels", DEFAULT_LABELS))
            if table.name == DEFAULT_LABELS or "--eval-labels" in values:
                continue
            ruler = table.with_name(DEFAULT_LABELS)
            if not (ruler.is_file() and table.is_file()):
                # SILENT WITHOUT THE TABLES, like `readings`: they are gitignored, so every clone and CI
                # would fail this check on the absence of data rather than on a defect. The moment this
                # rule exists for is arming, and the tables are on disk then.
                continue
            cells = Ruler(ruler).moved_cells(Ruler(table))
            if cells:
                moved[name] = (
                    f"--labels names {table.name} with no --eval-labels, and that table scores {cells} "
                    f"holdout cells differently from {DEFAULT_LABELS} — so its score is measured against "
                    f"a ruler no other run was measured against, and differs for that reason alone"
                )
        return moved

    def decode_smoke_caches(
        self,
        invocations: dict[str, str],
        refused: dict[str, str],
        first: int,
        workers: int = SMOKE_DECODE_WORKERS,
    ) -> list[str]:
        """Decode the TRUNCATED cache each accepted arm's reading names, where it is not there already.

        THE HALF THAT MAKES AN UNTRIED READING SMOKEABLE, and without it the smoke could only ever run
        arms whose full corpus was already decoded — so the one arm most in need of the check, a new
        reading axis nobody has run the memory point of, was exactly the one that could not get it. A
        full decode is two hours; sixty studies is under a minute, and it is a different file by name.

        Returns the keys built, which is what a caller logs — arms sharing a reading decode once.
        """
        built: list[str] = []
        for name, reading in self.readings(invocations).items():
            smoke = reading.truncated(first)
            if name in refused or smoke.decoded():
                continue
            logger.info("decoding %d studies at %s so the smoke has something to read", first, smoke.key)
            smoke.decode(workers)
            built.append(smoke.key)
        return built


def main() -> None:
    """Check every committed arm, and exit non-zero if any would be refused."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--arms", type=Path, default=ARMS)
    command.parser.add_argument("--stub", type=Path, default=STUB)
    command.parser.add_argument(
        "--require-caches",
        action="store_true",
        help="also refuse an arm whose cache is not decoded yet. Off by default because a queue may "
        "legitimately build one in an earlier slot; on when arming a queue that builds nothing.",
    )
    command.parser.add_argument(
        "--smoke",
        type=Path,
        default=None,
        help="also RUN every accepted arm for one epoch on --smoke-studies studies, writing throwaway "
        "checkpoints to this directory. Minutes rather than seconds. It WANTS the card, because it is "
        "the only check that sees a recipe fail on memory and no CPU run can answer that — but it does "
        "not require one: measured 2026-08-16, one whole-stack arm decodes and trains an epoch on CPU "
        "in about seven minutes, so a single arm can be smoked while the card belongs elsewhere. The "
        "full sweep cannot: that is over the per-arm timeout times thirty-five.",
    )
    command.parser.add_argument("--smoke-studies", type=int, default=SMOKE_STUDIES)
    command.parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="ARM",
        help="check (and smoke) only the named arms. The full smoke is thirty-five arms times an epoch "
        "on the card; a single arm about to be queued is two minutes. Names are the ones the arms "
        "print, and a name that matches no arm is refused rather than quietly selecting nothing.",
    )
    command.parser.add_argument(
        "--as-queued",
        nargs="+",
        default=None,
        metavar="ENTRY",
        help="check the arms AS A QUEUE WOULD RUN THEM, each entry being one `after.sh` argv element "
        "carrying its own arguments: --as-queued 'experiments/arms/resolution.sh 448'. Without this "
        "every check here runs the arm bare, so a non-default rung is queued on flags nothing parsed, "
        "cache-checked or smoked — the flag-typo-costs-a-slot failure, at the one invocation that is "
        "not the shipping one. Composes with --smoke.",
    )
    command.parser.add_argument(
        "--switches",
        action="store_true",
        help="also check the switches NO arm passes — one default recipe per uncovered option, from "
        "`switch_coverage`. Composes with --smoke, which is where it earns itself: five switches were "
        "landed-checked, stamped and never run, so a break in one surfaced at a queue slot.",
    )
    arguments = command.parsed()

    check = ArmCheck(arguments.arms, arguments.stub)
    printed = check.queued(arguments.as_queued) if arguments.as_queued else check.invocations()
    # MERGED BEFORE the selection, so `--only switch_tie` names one of these the way it names an arm.
    switches = SwitchCoverage().invocations(printed) if arguments.switches else {}
    invocations = check.selected({**printed, **switches}, arguments.only)
    refused = check.refused(invocations)
    # A SMOKE-ONLY RECIPE IS NOT UNCACHED. A switch recipe exists to be RUN at sixty studies, and
    # `decode_smoke_caches` builds exactly that truncation for it; no queue will ever ask for its full
    # corpus, so `--tie last` reported "nothing decoded at 336px_12sl_last" on every sweep and always
    # would have — and a check that cries about a recipe working as specified is one nobody reads by
    # the third sweep. It also left --require-caches unable to pass alongside --switches at all.
    uncached = check.uncached({name: flags for name, flags in invocations.items() if name not in switches})
    verdicts = check.verdicts(
        invocations, {**printed, **switches}, refused, uncached, require_caches=arguments.require_caches
    )
    # A LIST rather than a generator fed to `any`: that would stop reporting at the first refusal, and
    # every check here exists to be READ. What it returns is the refusal; what it is for is the log.
    refusing = [verdict.report(len(invocations), arming=bool(arguments.only)) for verdict in verdicts]
    broken: dict[str, str] = {}
    if arguments.smoke:
        # BEFORE the smoke, not as part of it: a smoke reads the truncated cache, and an arm on a
        # reading nothing has decoded would otherwise die on the memory map — which is the failure
        # the smoke exists to find, reported as though the recipe were at fault.
        check.decode_smoke_caches(invocations, refused, arguments.smoke_studies)
        broken = ArmSmoke(arguments.smoke_studies).reported(invocations, refused, arguments.smoke)
    for line in ArmShadow(check.arms).reported():
        logger.info("%s", line)
    logger.info("%d arms checked, %d refused, %d uncached", len(invocations), len(refused), len(uncached))
    Denominator(len(invocations), "no arm printed an invocation — the shell, not the arms, failed").refuse_empty()
    if any(refusing) or broken:
        sys.exit(1)


if __name__ == "__main__":
    main()
