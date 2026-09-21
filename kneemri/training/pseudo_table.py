"""A teacher's own predictions, written back as a label table: `python -m kneemri.training.pseudo_table`.

The half of transduction that is not a split. `--pseudo-fold` puts the holdout fold into TRAINING while
leaving it in the holdout for scoring; this writes the table that fold is trained UNDER. Together they
express the one thing no combination of existing flags could: train on the held-out studies with
invented labels, score them against the true ones.

WHY THIS IS NOT A LEAK, STATED PRECISELY. The pseudo rows carry the TEACHER's predictions, which the
teacher produced without ever seeing those studies' labels — the record is its fold-0 holdout. Nothing
derived from the true labels of those 909 studies enters training. Scoring still happens against
`report_labels_v2` through `--eval-labels`, so the ruler is untouched. An arm that trained on this table
and was scored WITHOUT that pin would be measuring agreement with its own teacher rather than with the
readings, which is the one way this module can produce a number that means nothing.

WHAT IT IS EXPECTED TO MEASURE, AND WHY THAT IS ~0 HERE. knee-tta1 measured naive self-training with a
LINEAR head under zero shift: 40 cells, none clearing +0.0005, and the delta shrank toward zero as the
pseudo-labelled tail grew — circularity measured, since pseudo-labels read off a ridge lie along that
ridge's own decision direction. Two suspects were named and only one is simulatable here: whether a
NONLINEAR head escapes that circularity. The other — that shift is zero by construction — cannot be
simulated, because our holdout is drawn from the same corpus. So the level this reads is an instrument
for suspect (1) and says nothing about what transduction would pay on the real test set.

THE CONFIDENCE COLUMNS ARE THE DESIGN DECISION, not a detail. `__conf` weights the loss, and the base
table's value is confidence in a REPORT READING that the teacher never performed. Carrying it over would
weight a model's guess by the certainty of a sentence, which is two different quantities sharing a
column. Every pseudo cell therefore gets ONE declared confidence, and it is a flag rather than a
constant so the arm that chooses it has to say so in its own line.

`__verdict` IS SET TO `PSEUDO` ON EVERY REPLACED CELL. It is prose that nothing reads, which is exactly
why it is the right place to make the table self-identifying: a pseudo table that got mistaken for a
reading would be undetectable from its numbers alone, since both live on [0, 1].
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.findings import STUDY, TARGETS
from kneemri.runs.holdout_record import HoldoutRecord

logger = logging.getLogger(__name__)

CONFIDENCE_SUFFIX = "__conf"
VERDICT_SUFFIX = "__verdict"
PSEUDO_VERDICT = "PSEUDO"
"""The token every replaced cell's verdict carries, so the table names itself in its own rows."""


class PseudoTable:
    """One base label table with a teacher's holdout predictions written over the studies it covers."""

    def __init__(
        self, base: pd.DataFrame, studies: list[str], predictions: Float[NDArray[np.float64], "studies labels"]
    ) -> None:
        self.base = base
        self.studies = studies
        self.predictions = predictions

    def covered(self) -> list[str]:
        """The teacher's studies that the base table actually carries, in the base's own order.

        Stated as a list rather than assumed, because a teacher record and a label table can disagree on
        the corpus — and a positional join between them would write one study's predictions onto another
        with no error at all. The denominator of every claim this module makes is `len(covered())`.
        """
        present = set(self.base[STUDY])
        return [study for study in self.studies if study in present]

    def written(self, confidence: float) -> pd.DataFrame:
        """The base table with the covered rows' twelve targets replaced by the teacher's predictions."""
        table = self.base.set_index(STUDY)
        ordered = pd.DataFrame(self.predictions, index=pd.Index(self.studies, name=STUDY), columns=list(TARGETS))
        covered = self.covered()
        for finding in TARGETS:
            table.loc[covered, finding] = ordered.loc[covered, finding]
            table.loc[covered, f"{finding}{CONFIDENCE_SUFFIX}"] = confidence
            table.loc[covered, f"{finding}{VERDICT_SUFFIX}"] = PSEUDO_VERDICT
        return table.reset_index()

    def coverage(self) -> list[str]:
        """How much of the teacher landed, and how far its predictions sit from the labels they replace."""
        covered = self.covered()
        truth = self.base.set_index(STUDY).loc[covered, list(TARGETS)].to_numpy(dtype=np.float32)
        ordered = pd.DataFrame(self.predictions, index=pd.Index(self.studies, name=STUDY), columns=list(TARGETS))
        drift = float(np.abs(ordered.loc[covered].to_numpy(dtype=np.float32) - truth).mean())
        return [
            f"{len(self.base)} base studies, {len(self.studies)} teacher studies, {len(covered)} written",
            f"{len(self.studies) - len(covered)} teacher studies absent from the base and skipped",
            f"mean |pseudo - true| over the replaced cells: {drift:.4f}",
        ]


def main() -> None:
    """Write a label table whose covered studies carry a teacher's predictions instead of their labels."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--base", type=Path, required=True, help="the true table whose rows lead")
    command.parser.add_argument("--record", type=Path, required=True, help="the teacher's .holdout.npz")
    command.parser.add_argument(
        "--confidence", type=float, required=True, help="the loss weight every pseudo cell carries"
    )
    command.parser.add_argument("--out", type=Path, required=True, help="where the pseudo table is written")
    arguments = command.parsed()

    reading = HoldoutRecord(arguments.record).read()
    if reading is None:
        message = f"{arguments.record} holds no predictions, so there is no teacher to write down"
        raise FileNotFoundError(message)
    pseudo = PseudoTable(pd.read_csv(arguments.base), reading.studies, reading.predictions)
    command.report(pseudo.coverage())

    destination = arguments.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pseudo.written(arguments.confidence)
    table.to_csv(destination, index=False)
    command.report([f"{len(table)} rows -> {destination}"])


if __name__ == "__main__":
    main()
