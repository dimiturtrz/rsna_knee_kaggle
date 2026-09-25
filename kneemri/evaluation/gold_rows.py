"""The radiologist-annotated rows of train.csv, and a report table read onto them.

WHY THIS IS ITS OWN MODULE. Eight modules across `kneemri/evaluation` ask for these two arrays, and
until 2026-08-29 they all reached through `SupervisionGap` for them — the gold rows lived inside one
consumer of the gold rows. That is a shared vocabulary in a leaf's class, and it cost a real cycle:
`annotated_readings` reads OUR members over the annotated studies and so needs the identifiers, while
`supervision_gap` needs `annotated_readings` to score a model we fitted, so each imported the other and
`graph.py --assert` blocked every commit touching the package.

It is deliberately free of torch. `annotated_readings` imports torch at module scope, so anything the
published path shares with it has to sit BELOW both or the published reading pays for a GPU import it
never uses.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Float

from core import cli
from core.findings import TARGETS

logger = logging.getLogger(__name__)

PRESENT = 0.5
"""Where a graded value becomes a positive, matching `LabelScores.per_label`'s own binarisation."""


class GoldRows:
    """The 58 studies a radiologist read on every finding, and any table put in their order."""

    def annotated(self, train_table: Path) -> tuple[list[str], Float[np.ndarray, "studies targets"]]:
        """The studies a radiologist annotated on EVERY finding, and those annotations.

        All-or-nothing rather than per-cell: train.csv annotates a study on all twelve or none, so a
        partial row would be a study whose columns mean different things, and there are none to
        scavenge. Returned in table order so a caller can align predictions to it.
        """
        table = pd.read_csv(train_table).set_index("StudyInstanceUID")
        columns = [target for target in TARGETS if target in table.columns]
        complete = table[columns].dropna(how="any")
        return list(complete.index), complete.to_numpy(dtype=np.float32)

    def reported(self, labels: Path, identifiers: list[str]) -> Float[np.ndarray, "studies targets"]:
        """The report-derived table for exactly these studies, in their order, READ RAW.

        Raw is the load-bearing word. `Targets.load` overwrites the derived label with the radiologist
        annotation wherever a study has one, which is right for training and would make any measurement
        against those annotations compare them with themselves — a perfect score, and a meaningless one.

        A study the table does not carry, or a finding the reader could not read, lands at PRESENT: the
        same 0.5 default the rest of the project uses, which binarises to absent.

        ALL TWELVE COLUMNS, ALWAYS, even ones the table does not carry. `SupervisionGap.against` names
        its columns by position — `TARGETS[column]` — so a table missing one finding would return eleven
        columns and every finding after the gap would be scored under its neighbour's name. Ours carries
        all twelve and the bug never fired; a foreign reader is not owed that assumption.
        """
        table = pd.read_csv(labels).set_index("StudyInstanceUID").reindex(identifiers)
        values = table.reindex(columns=list(TARGETS)).to_numpy(dtype=np.float32)
        return np.asarray(np.nan_to_num(values, nan=PRESENT), dtype=np.float32)


def main() -> None:
    """Write the annotated identifiers one per line, which is what `build_cache --only-studies` reads.

    A cache restricted to these studies is how a feed question gets asked at 58 studies instead of 4,407:
    `foreign_gate` scores a published checkpoint against its own `gold_auc` on exactly this set, so a
    depth or band the full corpus cannot afford to cache is affordable here at a thousandth of the bytes.
    """
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--out", type=Path, required=True, help="where to write the identifiers")
    arguments = command.parsed()

    identifiers, _ = GoldRows().annotated(command.rooted(arguments).train_table())
    arguments.out.write_text("\n".join(identifiers) + "\n", encoding="utf-8")
    logger.info(f"wrote {len(identifiers)} annotated identifiers to {arguments.out}")


if __name__ == "__main__":
    main()
