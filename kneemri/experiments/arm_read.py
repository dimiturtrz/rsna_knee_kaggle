"""Read a finished arm the way its own header said to: per label, against the control it declared.

`python -m kneemri.experiments.arm_read --arm cover_labelwise --labels <table>`

THE DEFECT THIS EXISTS TO COVER, and it happened on the day this module was written. `stack224` finished
at a twelve-label mean of 0.8476 against `native224`'s 0.8502 — a mean delta of -0.0026, which reads as a
flat null and would have closed the slice-coverage axis. Its own header had pre-registered the opposite
reading: "a flat mean with Lateral Meniscus and Fracture both up is the finding". Reading it per label
showed a two-label collapse (-0.0263, -0.0180) inside a nine-label rise, which names an AGGREGATOR
failure rather than a dead idea, and the arm that followed exists only because someone remembered to look.

Nothing in the tree made that happen. `run_comparison` has computed per-label deltas with their own error
bars for weeks, `ArmHeader.controls` has parsed `# control  <name> = <score>` for as long, and the two
had never been joined — so the comparison was hand-rolled from a metrics query every time, which means it
was run when someone thought of it. An instrument that is optional is an instrument that is skipped in
exactly the case that most needed it: the arm whose mean looks decided.

WHAT IT DOES. Resolves the arm's declared control out of its shell header and hands both to
`RunComparison`, which is the tool that already knows how to print the per-label table. It computes
nothing of its own and deliberately owns no arithmetic — a second implementation of a delta is a second
thing to keep true, and the one that disagreed would be the one quoted.

WHAT IT REFUSES. `--assert` exits non-zero for an arm that declares no control at all. That is not
pedantry about a comment: an arm with no control has no denominator, and every reading of it is a
comparison against whichever number the reader remembered. Seventy arms predate the convention and name a
CHECKPOINT rather than an arm, which this tool passes through untouched — `RunComparison.resolved` takes
either. Silence is the defect; a checkpoint is a control.

IT DOES NOT JUDGE. No branch, no verdict, no keep/kill. The arm's header holds the pre-registered bands
and a tool that scored them against its own table would be pre-registering after the number. This prints
the table the header asked to be read.
"""

import logging
import re
from collections.abc import Iterable
from pathlib import Path

from core import cli
from kneemri.evaluation.run_comparison import RunComparison
from kneemri.experiments.arm_header import ArmHeader
from kneemri.runs.run_record import METRICS_SUFFIX, RunRecord

logger = logging.getLogger(__name__)

DECLARED_TOLERANCE = 5e-4
"""How far a run's best epoch may sit from the score its header quoted and still be that run.

Headers quote four decimals, so the rounding alone is 5e-5; the rest of the width is for a number
copied off a report that printed a different epoch's rounding. Wide enough to survive transcription,
far narrower than the 0.0062 report-holdout floor, so it can never merge two genuinely different runs.
"""

OUT_FLAG = re.compile(r'--out\s+"?([^"\s]+)"?')
"""What an arm writes, read off its own invocation. The quotes are optional because an arm may spell the
path either way, and `$PROCESSED/` expands to nothing here — only the stem is wanted."""


class ArmRead:
    """An arm name, the control its header declares, and the per-label table joining them."""

    def __init__(self, headers: ArmHeader | None = None) -> None:
        self.headers = headers if headers is not None else ArmHeader()

    def control(self, arm: str) -> str | None:
        """The control this arm declares, or None where its header names none.

        A single-arm read of `ArmHeader.controls` rather than a second parse, so the line this tool
        requires is the same line `arm_check` reports on and there is one regex in the tree for it.
        """
        return self.headers.controls([arm]).get(arm)

    def produced(self, arm: str) -> str:
        """The run name this arm's `--out` writes, which is NOT the arm's own name.

        An arm is a shell called `cover_labelwise.sh` and it writes `cover_labelwise_s2026.pt`, so the
        metrics beside it are `cover_labelwise_s2026.metrics.jsonl`. This tool resolved the ARM name and
        reported `cover_labelwise.metrics NOT READ: does not exist` — a real, finished, converged run
        reading as a missing one, for every arm in the tree whose checkpoint carries the suffix, which is
        nearly all of them. Passing the checkpoint name instead then fails the other way, because the
        HEADER is keyed on the shell. Both names are needed and they are read from their own sources.

        Falls back to the arm name when no `--out` is parsed, which is the pre-suffix convention and the
        spelling this module's own docstring was written against.
        """
        found = OUT_FLAG.search(self.headers.text(arm))
        return Path(found.group(1)).stem if found else arm

    def sibling(self, named: str, declared: float | None, models: Path) -> Path:
        """The run a control NAME points at, picked out of its split-suffixed siblings by its own score.

        A header names an ARM and the disk holds RUNS. `lr4e-5_ws` — described in three headers as "the
        control every other member reads" — has no checkpoint of that name at all; it has
        `lr4e-5_ws_cf`, `lr4e-5_ws_ff` and `lr4e-5_ws_s2026`, and those are three different splits, one
        of which has seen every holdout study. So the name resolved to a path that could not exist and
        the tool reported the project's most-used control as a missing run. `produced` cannot help here:
        it reads the `--out` of a shell, and there is no `lr4e-5_ws.sh` in the tree to read.

        THE DECLARED SCORE IS WHAT DISAMBIGUATES, and it must be — guessing the suffix would pick a
        split by convention and read a fold-0 arm against a full-fit control without saying so. A
        sibling matching the number the header wrote down is the run whose score the header quoted, by
        construction. Ambiguity and silence both raise, naming every candidate and its own best epoch,
        because a control resolved wrongly is worse than one not resolved at all.

        THE CHECKPOINT ALONE SATISFIES THE DIRECT BRANCH, not just a metrics file. `wsfold0_e6` — the
        control knee-bot's whole argument rests on — predates the recorder and kept its predictions
        rather than epoch rows, which `RunComparison.scored` reads and `RunRecord.exists` does not. On
        the metrics test alone it fell through to the sibling search, found no siblings either, and a
        control that resolves perfectly one layer down was reported as matching nothing.
        """
        direct = RunComparison.resolved(Path(self.produced(named)), models)
        if direct.exists() or RunRecord(direct).exists():
            return direct
        records = models.glob(f"{named}_*{METRICS_SUFFIX}.jsonl")
        runs = [models / f"{record.name.removesuffix(f'{METRICS_SUFFIX}.jsonl')}.pt" for record in records]
        best = ((run, RunRecord(run).best()) for run in runs)
        scored = {run: row["auc"] for run, row in best if row is not None}
        if declared is not None:
            matched = [path for path, auc in scored.items() if abs(auc - declared) <= DECLARED_TOLERANCE]
            if len(matched) == 1:
                return matched[0]
        candidates = " ".join(f"{path.stem}={auc:.4f}" for path, auc in sorted(scored.items())) or "none"
        message = f"control {named} = {declared} matches no single run under {models}; siblings: {candidates}"
        raise SystemExit(message)

    def uncontrolled(self, arms: Iterable[str]) -> list[str]:
        """The arms declaring no control — the ones whose every reading is against a remembered number.

        PURE, and separate from `control` for the reason `arm_check.unpriced` is separate from the `bd`
        call beside it: the unit gate refuses a test that touches a data root, and this half holds the
        judgement while the header text it reads comes from a directory the caller chooses.
        """
        return sorted(arm for arm in arms if self.control(arm) is None)

    def arms(self) -> list[str]:
        """Every arm the shell directory holds, by invocation name.

        The census is what makes `--assert` worth running: one arm's silence is a note to its author,
        and the fraction of the corpus that is silent is a fact about the harness.
        """
        directory = self.headers.arms
        return sorted(shell.stem for shell in directory.glob("*.sh"))

    def report(self, arm: str, models: Path, labels: Path | None) -> str:
        """The per-label table for one arm against its declared control, error bars included.

        Raises rather than printing a mean when the arm declares no control. The whole point of the
        module is that a mean is what gets read when nothing forces the alternative, so falling back to
        one on the failure path would reinstate the defect at exactly the moment it bites.

        The root is the MODELS root, which is where `run_train` writes every `.metrics.jsonl`. It was
        the PROCESSED root until 2026-09-03, and that is not a near miss: no run's metrics have ever
        been written under processed, so every name this tool was given resolved to a path that could
        not exist and came back as `control <name>.metrics.jsonl does not exist`. That reads as a
        missing RUN rather than as a wrong ROOT, which is why the tool was believed to be waiting on
        arms that declare a control rather than on this line. `run_comparison`, which does the same
        resolution one layer down, passes `root.models()` and always did.
        """
        named = self.control(arm)
        if named is None:
            message = f"{arm} declares no `# control  <name> = <score>` line, so it has no denominator"
            raise SystemExit(message)
        control = self.sibling(named, self.headers.control_scores([arm]).get(arm), models)
        run = RunComparison.resolved(Path(self.produced(arm)), models)
        return RunComparison(control, [run], labels, None).report()


def main() -> None:
    """Read one arm per label: `python -m kneemri.experiments.arm_read --arm <name> --labels <table>`."""
    command = cli.Command(ArmRead.__doc__, logger)
    command.parser.add_argument("--arm", help="the arm's invocation name, as run_train printed it")
    command.parser.add_argument("--labels", type=Path, help="the label table, to print each delta's own error bar")
    command.parser.add_argument(
        "--assert",
        dest="asserted",
        action="store_true",
        help="exit non-zero when the arm declares no control",
    )
    arguments = command.parsed()
    read = ArmRead()
    if arguments.asserted:
        subjects = [arguments.arm] if arguments.arm else read.arms()
        silent = read.uncontrolled(subjects)
        logger.info("%d of %d arms declare no control: %s", len(silent), len(subjects), " ".join(silent))
        if silent and arguments.arm:
            raise SystemExit(f"{arguments.arm} declares no control, so no reading of it has a denominator")
        return
    if arguments.arm is None:
        raise SystemExit("--arm names the arm to read; only --assert may sweep every arm at once")
    root = command.rooted(arguments)
    logger.info("%s", read.report(arguments.arm, root.models(), arguments.labels))


if __name__ == "__main__":
    main()
