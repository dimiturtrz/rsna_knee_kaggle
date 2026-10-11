"""Did a foreign model train on our holdout? `python -m kneemri.evaluation.foreign_overlap`.

WHY IT EXISTS. On 2026-08-27 six published `raptor` checkpoints were scored on our 909-study fold-0
holdout and the best of them read 0.8938 against our own best member's 0.8634. That `+0.0304` became
the measured support for a "member gap" driving six P0s. On 2026-08-28 the same checkpoint scored
**0.923 solo on the board, exactly level with our own best single** — a five-floor offline lead
delivering nothing, against a `board_transfer` that has never compressed anything to zero.

The reason is one set intersection, and it was available the whole time. Their published label table
covers 4349 of our 4407 studies: the corpus minus the 58 gold and nothing else. **896 of our 909
holdout studies are in it**, the 13 absent being exactly the gold studies that fall in fold 0. Their
number was 98.6% in-sample; ours was held out. On the two rulers where BOTH models are honest — the
gold 58 and the board — they tie.

WHAT THIS REFUSES, AND WHY A REFUSAL. A contaminated score is not wrong-looking on the page. It is a
plausible number in the right range that arrives with a fallback rate and a per-finding table, and it
outranks a forum rumour precisely because it is measured. Nothing downstream can tell it apart, so the
guard has to fire where the ruler is chosen rather than where the number is read.

THE DEFAULT IS GUILT. A competitor's model has seen our holdout unless their own artefacts prove
otherwise: our fold assignment is ours, it is a hash of report text, and no external party has any
reason to respect it. So a foreign checkpoint is priced on the gold 58, on the board, or not at all.

WHAT IT CANNOT SEE. A published label table is evidence about the rows they had, not proof of the rows
they trained on. A table whose row set is exactly "the corpus minus the oracle" is as strong as that
evidence gets — it is the shape of holding out the oracle only, and here the checkpoints' own
self-reported metric is `gold_auc` on those 58 — but a model trained on a subset of its own table
would still trip this. That is the correct direction to be wrong in.
"""

import logging
from pathlib import Path

import pandas as pd

from core import cli
from core.data.corpus.folds import ReportFolds
from core.findings import STUDY
from core.paths import DataRoot

logger = logging.getLogger(__name__)

IDENTIFIER = STUDY
"""The join key every table in this project carries, ours and theirs alike."""

CLEAN = 0
"""No holdout study appears in the foreign row set, so our ruler is honest for their model."""


class ForeignOverlap:
    """How much of one of our fold holdouts a foreign artefact's row set already contains."""

    def holdout(self, train_table: Path, fold: int) -> list[str]:
        """The studies in one fold's holdout, by the same assignment every member of ours was fitted on.

        Read from `train.csv` rather than from a member's recorded corpus so the answer does not depend
        on which checkpoint is being questioned — the fold is a property of the data, and a foreign
        model has no checkpoint of ours to read it from.
        """
        identifiers = list(pd.read_csv(train_table)[IDENTIFIER])
        folds = ReportFolds()
        _, held = folds.holdout(folds.assign(str(train_table), identifiers), fold)
        return [identifiers[index] for index in held]

    def covered(self, holdout: list[str], foreign: Path) -> list[str]:
        """Which of those studies the foreign table carries a row for, in holdout order.

        Order is kept rather than sorted so a caller printing the first few gets the same few every
        run, which is what makes a spot-check against their file reproducible.
        """
        rows = set(pd.read_csv(foreign)[IDENTIFIER])
        return [name for name in holdout if name in rows]

    def verdict(self, covered: int, total: int) -> list[str]:
        """The reading, and it has three outcomes rather than a threshold on a percentage.

        NONE is the only clean one. ALL-BUT-A-FEW is the shape that actually occurred and is worse than
        a partial overlap, because the few they lack are the oracle studies — that row set says they
        withheld a validation set and kept everything else, which is exactly the model whose score on
        our holdout means nothing.
        """
        if covered == CLEAN:
            return [f"CLEAN: none of our {total} holdout studies are in their row set — our ruler is honest here"]
        share = covered / total
        return [
            f"CONTAMINATED: {covered} of {total} holdout studies ({share:.1%}) are in their row set",
            "  their score on this holdout is in-sample and ours is not; the comparison is void",
            "  price them on the gold 58 or on the board, and re-read any seat value computed here",
        ]


def main() -> None:
    """Report how much of a fold holdout a foreign label table or corpus listing already contains."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--foreign", type=Path, required=True, help=f"a table with a {IDENTIFIER} column")
    command.parser.add_argument("--fold", type=int, default=0, help="which holdout their score would be read on")
    command.parser.add_argument("--assert-clean", action="store_true", help="exit non-zero on any overlap")
    arguments = command.parsed()

    root: DataRoot = command.rooted(arguments)
    overlap = ForeignOverlap()
    holdout = overlap.holdout(root.train_table(), arguments.fold)
    covered = overlap.covered(holdout, arguments.foreign)
    header = f"{arguments.foreign.name} vs fold-{arguments.fold} holdout"
    command.report([header, *overlap.verdict(len(covered), len(holdout))])
    if arguments.assert_clean and covered:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
