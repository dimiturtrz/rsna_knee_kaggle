"""How many series hold slices of DIFFERENT matrix sizes? `python -m core.data.corpus.diagnose_matrix`.

`SlotReader.cropped` cannot stack planes of unequal size, so it zeroes the odd ones out. Which planes
are "odd" used to be decided by the FIRST sampled slice, so a series whose first sampled slice was the
minority had every other slice blackened while the presence mask still reported the slot acquired. The
majority now decides, and this is the measurement that says how much that was ever worth: the rate is
over SERIES READ, not over series that happen to be mixed, because a rate computed inside the subset
that shows the effect is not a rate.

Headers only — `Rows`/`Columns` with the pixels left on disk — so this costs minutes, not a cache
rebuild.

Measured 2026-08-16 on the first 300 train studies: **0 of 1682 series** hold more than one matrix
size. The defect was real and the fix stands as insurance for the hidden test, but on this training
corpus it never fired — no member on disk was ever served a blackened slot for this reason.
"""

import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pydicom

from core import cli

logger = logging.getLogger(__name__)

SAMPLED_STUDIES = 300
"""Studies read by default. The question is a RATE, and a few hundred studies is some thousands of
series — enough to separate "one series in a thousand" from "one in ten", which is the decision this
measurement feeds."""


class MatrixAudit:
    """Counts, per series, how many distinct slice matrix sizes it holds."""

    @staticmethod
    def survey_one(directory: str) -> list[dict[str, object]]:
        """One study: a row per series, with its shape mix."""
        rows: list[dict[str, object]] = []
        for series in sorted(Path(directory).iterdir()):
            if not series.is_dir():
                continue
            shapes: Counter[tuple[int, int]] = Counter()
            for slice_file in sorted(series.iterdir()):
                try:
                    header = pydicom.dcmread(str(slice_file), stop_before_pixels=True, force=True)
                    shapes[(int(header.Rows), int(header.Columns))] += 1
                except (OSError, ValueError, AttributeError, TypeError):
                    continue
            if not shapes:
                continue
            majority = shapes.most_common(1)[0][1]
            rows.append(
                {
                    "study": Path(directory).name,
                    "series": series.name,
                    "slices": sum(shapes.values()),
                    "distinct_shapes": len(shapes),
                    "minority_slices": sum(shapes.values()) - majority,
                }
            )
        return rows

    def run(self, source: Path, identifiers: list[str], workers: int) -> pd.DataFrame:
        """Every study in parallel; only headers are read."""
        directories = [str(source / identifier) for identifier in identifiers]
        rows: list[dict[str, object]] = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for done, study in enumerate(pool.map(self.survey_one, directories, chunksize=4), start=1):
                rows.extend(study)
                if done % 100 == 0:
                    logger.info(f"  surveyed {done}/{len(directories)} studies")
        return pd.DataFrame(rows)


def main() -> None:
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--out", type=Path)
    command.parser.add_argument("--studies", type=int, default=SAMPLED_STUDIES)
    command.parser.add_argument("--workers", type=int, default=20)
    arguments = command.parsed()

    from core.data.corpus.study_cache import CorpusSplit  # noqa: PLC0415 - CLI-only, as above

    split = CorpusSplit(command.rooted(arguments).raw())
    identifiers = split.identifiers()[: arguments.studies]
    table = MatrixAudit().run(split.source, identifiers, arguments.workers)
    if arguments.out:
        table.to_csv(arguments.out, index=False)

    mixed = table[table.distinct_shapes > 1]
    logger.info(f"\n{len(table)} series over {len(identifiers)} studies; {len(mixed)} hold more than one matrix size")
    if len(table):
        logger.info(f"mixed rate over series read: {len(mixed) / len(table):.4f}")
    if len(mixed):
        logger.info(f"minority slices in a mixed series: median {mixed.minority_slices.median():.0f}")
        logger.info("%s", mixed.distinct_shapes.value_counts().to_dict())


if __name__ == "__main__":
    main()
