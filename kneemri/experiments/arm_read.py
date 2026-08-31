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
from collections.abc import Iterable
from pathlib import Path

from core import cli
from kneemri.evaluation.run_comparison import RunComparison
from kneemri.experiments.arm_header import ArmHeader

logger = logging.getLogger(__name__)


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

    def report(self, arm: str, processed: Path, labels: Path | None) -> str:
        """The per-label table for one arm against its declared control, error bars included.

        Raises rather than printing a mean when the arm declares no control. The whole point of the
        module is that a mean is what gets read when nothing forces the alternative, so falling back to
        one on the failure path would reinstate the defect at exactly the moment it bites.
        """
        named = self.control(arm)
        if named is None:
            message = f"{arm} declares no `# control  <name> = <score>` line, so it has no denominator"
            raise SystemExit(message)
        control = RunComparison.resolved(Path(named), processed)
        return RunComparison(control, [RunComparison.resolved(Path(arm), processed)], labels, None).report()


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
    logger.info("%s", read.report(arguments.arm, root.processed(), arguments.labels))


if __name__ == "__main__":
    main()
