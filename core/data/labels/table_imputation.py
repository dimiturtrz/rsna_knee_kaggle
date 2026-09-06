"""Fill a finding the reports are SILENT about from one they state: `python -m core.data.labels.table_imputation`.

WHY A SILENT CELL IS NOT A NEGATIVE CELL, which is the whole defect. `report_reader` writes three verdicts
per finding — YES, NO and UNK — and the UNK cells get a low prior value because the report did not raise
the subject. For eleven findings that is close enough to right: a knee report that never mentions the ACL
is usually a knee whose ACL is intact. For Synovitis it is wrong by construction: 3713 of 4407 studies
(84.2%) are UNK, and the radiologist panel calls 46.6% of studies synovitic. A prior applied to 84% of the
corpus cannot separate half of it.

WHAT THE CONFUSION TABLE SAYS, and it is why the fix is DIRECTIONAL rather than a re-threshold.
`supervision_gap --labels report_labels_v2.csv` reads Synovitis at 9.7% false positive over the gold
negatives and 51.9% FALSE NEGATIVE over the gold positives. The reader is not miscalibrated, it is
ABSTAINING: it says no when the report says nothing, and the studies it abstains on are where the
radiologist says yes. Every candidate blend on disk reproduces those two rates to the digit, because none
of them re-read the silence — they average readers that all inherited it.

WHY THE SOURCE IS EFFUSION AND NOT A PRIOR. Radiological convention writes synovial thickening into the
effusion sentence: a report that describes a joint effusion has described the compartment synovitis lives
in, and one that says nothing about either has looked at neither. So the effusion column is a reading of
the same sentence, taken by the same reader, on the same study — not an outside guess. `stevenleehans`
published this exact rule for the v2 tables and reported gold Synovitis 0.678 -> 0.790.

THE RULE ONLY EVER FILLS SILENCE. A cell whose verdict is YES or NO was stated by the report and is left
exactly as it was, so the imputation can never overrule the reader — it can only reach the cells the reader
declined to answer. That is what keeps this a supervision fix rather than a second reader: on the cells
where both exist, this module writes nothing at all.

AND IT IS ONE TREATMENT PER TABLE, DECLARED. The target and source are arguments rather than a table of
pairs, because every additional pair is a claim about radiological convention that has to be argued and
measured on the gold 58 separately. Synovitis-from-Effusion is the one with a published mechanism and a
published number; the next pair starts at zero evidence, not at this one's.
"""

import logging
from pathlib import Path

import pandas as pd

from core import cli
from core.data.labels.reader_agreement import SILENT, VERDICT_SUFFIX
from core.findings import TARGETS

logger = logging.getLogger(__name__)

CUT = 0.5
"""The cut a census counts across — the same 0.5 every confusion table on this project is read at."""

IMPUTED = "IMPUTED"
"""The verdict written onto a filled cell, so a later reader of the table can tell it from a stated one.

A filled cell that kept UNK would be indistinguishable from an untouched one, and a filled cell relabelled
YES would claim the report said something it did not. It is a third thing and it is spelled as one."""


class SilenceImputation:
    """One label table, with the cells a reader declined to answer filled from a finding it did answer."""

    def __init__(self, table: pd.DataFrame) -> None:
        self.table = table

    def undecided(self, target: str) -> "pd.Series[bool]":
        """The rows whose verdict for this finding is silence — the only rows this module may write.

        A table with no verdict column for the finding yields an all-False mask rather than an error: the
        caller then sees `0 of N cells filled` and knows the table cannot carry the treatment, which is a
        readable answer. Guessing which cells were silent from the VALUES would be the same abstention
        defect one layer up.
        """
        column = f"{target}{VERDICT_SUFFIX}"
        if column not in self.table.columns:
            return pd.Series(data=False, index=self.table.index)
        return self.table[column] == SILENT

    def imputed(self, target: str, source: str) -> pd.DataFrame:
        """A copy of the table with `target`'s silent cells carrying `source`'s value for the same study."""
        filled = self.table.copy()
        silent = self.undecided(target)
        filled.loc[silent, target] = self.table.loc[silent, source]
        verdict = f"{target}{VERDICT_SUFFIX}"
        if verdict in filled.columns:
            filled.loc[silent, verdict] = IMPUTED
        return filled

    def census(self, target: str, source: str) -> list[str]:
        """How many cells the fill reaches and what it does to them, because the count IS the reading.

        A fill that moved three cells and a fill that moved four thousand are different treatments with
        the same name, and the mean before and after says which side of the cut the new mass lands on.
        """
        silent = self.undecided(target)
        before = self.table.loc[silent, target]
        after = self.table.loc[silent, source]
        stated = int((~silent).sum())
        filled = self.imputed(target, source)[target]
        return [
            f"{target} <- {source}: {int(silent.sum())} of {len(self.table)} cells silent, "
            f"{stated} stated and untouched",
            f"  filled cells: mean {before.mean():.4f} -> {after.mean():.4f}",
            f"  of them above {CUT}: {int((before > CUT).sum())} -> {int((after > CUT).sum())}",
            f"  whole column mean: {self.table[target].mean():.4f} -> {filled.mean():.4f}",
        ]


def main() -> None:  # devtools-ignore: test-mirror
    """Write a label table with one finding's silent cells filled from another finding's reading."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--table", type=Path, required=True, help="the label table to fill")
    command.parser.add_argument("--target", default="Synovitis", choices=TARGETS, help="the silent finding")
    command.parser.add_argument("--source", default="Effusion", choices=TARGETS, help="the finding it is read from")
    command.parser.add_argument("--out", type=Path, required=True, help="where the filled table is written")
    arguments = command.parsed()

    imputation = SilenceImputation(pd.read_csv(arguments.table))
    command.report(imputation.census(arguments.target, arguments.source))
    filled = imputation.imputed(arguments.target, arguments.source)
    out: Path = arguments.out
    out.parent.mkdir(parents=True, exist_ok=True)
    filled.to_csv(out, index=False)
    command.report([f"{len(filled)} rows -> {out}"])


if __name__ == "__main__":
    main()
