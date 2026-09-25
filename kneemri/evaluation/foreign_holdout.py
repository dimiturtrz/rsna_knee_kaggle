"""A foreign trainer's out-of-fold arrays, rewritten as one of this repo's holdout records.

WHY A CONVERSION AND NOT A READER. His trainer writes `raptor_oof_{tag}_fold{fold}.npz` with the keys
`pred`, `truth`, `ids`, `fold`, `nfolds`. Every instrument here — `run_comparison`, `seat_value`,
`roster_check`, the paired floor — reads `HoldoutRecord`, and none of them will ever learn his spelling.
Teaching each of them a second format is how a project ends up with two rulers; converting once at the
edge leaves exactly one, and the converted file is a record like any other from that point on.

HIS TRUTH IS NOT KEPT, AND THAT IS THE WHOLE POINT. The `truth` array in his npz is HIS soft label table,
so scoring his predictions against it asks whether his model agrees with the reader that taught it — a
number that is high for a well-fitted model and says nothing about the findings. Our truth is joined onto
his study identifiers instead, which makes his run and ours answerable by the same question. Where the
two tables differ is itself a live cell (`foreign_labels_dense`); it cannot also be the ruler.

THE ROWS ARE JOINED BY IDENTIFIER, NEVER BY POSITION. His fold file and ours are the same 909 studies but
nothing guarantees the same order, and a positional join would produce a full-length array of every
study scored against another study's truth — which is a plausible AUC, in the 0.5-0.7 band a real result
lives in, with nothing anywhere to mark it wrong.

`source` is FOREIGN, so a reader can never mistake these arrays for ones this repo's trainer produced.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from kneemri.evaluation.gold_rows import GoldRows
from kneemri.runs.holdout_record import FOREIGN, UNKNOWN_EPOCH, HoldoutRecord

logger = logging.getLogger(__name__)

PREDICTIONS = "pred"
"""His key for the out-of-fold predictions. Ours is `predictions`, which is why this module exists."""

IDENTIFIERS = "ids"


@dataclass(frozen=True)
class ForeignHoldout:
    """One foreign out-of-fold file, and the label table its predictions are to be judged against."""

    source: Path
    labels: Path

    def read(self) -> tuple[list[str], Float[NDArray[np.float64], "studies labels"]]:
        """His identifiers and predictions, refusing a file that does not carry both.

        Refused rather than defaulted: an OOF file missing its identifiers cannot be joined to anything,
        and the only remaining option would be the positional join this module exists to prevent.
        """
        held = np.load(self.source, allow_pickle=True)
        missing = [key for key in (PREDICTIONS, IDENTIFIERS) if key not in held]
        if missing:
            message = f"{self.source} carries no {missing} - it is not an out-of-fold file"
            raise ValueError(message)
        return [str(identifier) for identifier in held[IDENTIFIERS]], held[PREDICTIONS].astype(np.float64)

    def written(self, checkpoint: Path) -> Path:
        """His predictions beside `checkpoint`, scored-ready against OUR labels, as a holdout record.

        Named for the CHECKPOINT rather than for his file, because that is the name every instrument
        here resolves a run by — `run_comparison --arms <name>` finds a record by the checkpoint it
        sits beside, and a record under his own filename would be invisible to all of them.
        """
        identifiers, predictions = self.read()
        truth = GoldRows().reported(self.labels, identifiers)
        HoldoutRecord(checkpoint).write(identifiers, predictions, truth, UNKNOWN_EPOCH, source=FOREIGN)
        logger.info(
            "%d studies from %s, scored against %s, written beside %s",
            len(identifiers),
            self.source.name,
            self.labels.name,
            checkpoint.name,
        )
        return checkpoint


def main() -> None:
    """Convert one foreign OOF file: `python -m kneemri.evaluation.foreign_holdout --oof x.npz --out y.pt`."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--oof", type=Path, required=True, help="his out-of-fold npz")
    command.parser.add_argument("--out", type=Path, required=True, help="the checkpoint the record belongs to")
    command.parser.add_argument("--labels", type=Path, help="the truth table to judge him by; ours by default")
    arguments = command.parsed()

    root = command.rooted(arguments)
    labels = arguments.labels or root.processed() / "labels" / "report_labels_v2.csv"
    ForeignHoldout(source=arguments.oof, labels=labels).written(arguments.out)


if __name__ == "__main__":
    main()
