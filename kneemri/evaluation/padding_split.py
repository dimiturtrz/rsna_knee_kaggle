"""Whether a member's decode toll concentrates in the studies whose field it had to PAD out to.

THE QUESTION. Two of the pool's members are trained at a 160 mm extent and read a lower decode toll than
the eleven-member pool mean (`crop160_span` -0.0265 against -0.0336, sd 0.0038). That was charged to the
EXTENT. But only about 40% of series are physically wide enough to fill 160 mm, so on the other 60% the
requested field is not tissue, it is pad — and the composed failure mode D269 (c) names (wide fixed field
-> pad-filled borders -> pad learned as a site cue -> transfer toll) predicts the advantage should live in
one of those subsets and not the other.

WHY IT COSTS NO CARD. The metric is an unweighted mean of twelve rank-based AUCs, so it has no per-study
value and a subset cannot be recovered from a mean that has already been taken. `foreign_readout
--per-study` writes both shifted passes per study once; from there every split of those same studies is
arithmetic. This module is that arithmetic and nothing else — it trains nothing, decodes nothing, and
loads no encoder.

THE PARTITION IS FIXED ACROSS MEMBERS, at one extent, deliberately. Each member has its own `crop_mm` and
so its own padding, but splitting each member on its own extent would compare four different subsets and
answer nothing. Splitting every member on the SAME 160 mm question makes the narrower member the control:
on the studies that a 160 mm field must pad, a 130 mm member is padding nothing, so if the wide members'
advantage survives there it was never about the pad.

THE INTERVAL IS PRINTED WITH EVERY NUMBER because the subsets are half the holdout each and the paired
floor this project quotes (0.0062) was measured on all 909. A subset delta inside its own interval is not
a reading, and this module refuses to present one as though it were.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from core import cli
from core.data.corpus.study_metadata import IDENTIFIER
from kneemri.evaluation.scores import LabelScores

logger = logging.getLogger(__name__)

FILL_MM = 160.0
"""The extent the partition asks about — the field the two wide members were trained at."""

KEPT_SUFFIX = "::kept"
"""How `PerStudyDump` names a member's own-pixels pass; the member name is what precedes it."""

FOREIGN_SUFFIX = "::foreign"
"""And the second-decode pass. Both suffixes live here because a reader must not guess a column order."""

FIELD_COLUMN = "field_of_view_mm"
"""The per-study extent, written by `build_cache --metadata-only`: the NARROWEST side any series answered
with, because a square crop is bounded by whichever side runs out first."""


@dataclass(frozen=True)
class SubsetToll:
    """One member's decode toll over one subset of studies, with the size that produced it."""

    label: str
    studies: int
    kept: float
    foreign: float
    interval: tuple[float, float]

    @property
    def toll(self) -> float:
        """What the second decode cost on these studies — negative is a loss, as everywhere else."""
        return self.foreign - self.kept

    def logged(self) -> str:
        """The toll beside its n and its own interval, so a subset cannot be read as if it were the pool."""
        low, high = self.interval
        return (
            f"{self.label:>8} n={self.studies:<4} kept {self.kept:.4f} foreign {self.foreign:.4f} "
            f"toll {self.toll:+.4f} [{low:.4f}, {high:.4f}]"
        )


@dataclass(frozen=True)
class Coverage:
    """Which studies reach a requested extent, and which ones the metadata could say anything about.

    TWO masks rather than one, because `not wide enough` and `not known` are different facts and the
    obvious single-mask spelling silently merges them: `nan >= 160` is False, so an unanswered study
    would enter the PADDED subset as though its field had been measured and found short. That is the
    denominator error this project has now made three times, in the shape a boolean makes it easy to
    make.
    """

    fills: NDArray[np.bool_]
    known: NDArray[np.bool_]

    @classmethod
    def measured(cls, studies: list[str], metadata: pd.DataFrame, extent: float) -> "Coverage":
        """Per study: does its narrowest physical field reach `extent`, and did anything answer at all?"""
        widths = metadata.set_index(IDENTIFIER)[FIELD_COLUMN]
        answered = np.array([float(widths.get(study, np.nan)) for study in studies], dtype=np.float64)
        known = ~np.isnan(answered)
        if not known.all():
            missing = int((~known).sum())
            logger.warning("%d of %d studies have no %s and are in NEITHER subset", missing, len(studies), FIELD_COLUMN)
        return cls(fills=known & (answered >= extent), known=known)

    @property
    def padded(self) -> NDArray[np.bool_]:
        """Measured, and measured short — never merely unanswered."""
        return self.known & ~self.fills


class PaddingSplit:
    """The two subset tolls for one member, over a partition fixed before any member is read."""

    def __init__(self, scores: LabelScores, truth: NDArray[np.float32], coverage: Coverage) -> None:
        """`coverage` carries both masks: reaching the extent, and having been measured at all."""
        self.scores = scores
        self.truth = truth
        self.coverage = coverage

    @staticmethod
    def members_of(keys: "Iterable[str]") -> list[str]:
        """Every member the dump holds BOTH passes for, in the order they were written.

        Takes the KEYS rather than the file so the naming convention can be tested without an npz on
        disk — the convention is the thing that can break, and it is a string rule, not a file format.
        """
        return [key[: -len(KEPT_SUFFIX)] for key in keys if key.endswith(KEPT_SUFFIX)]

    def subset(
        self, label: str, kept: NDArray[np.float64], foreign: NDArray[np.float64], mask: NDArray[np.bool_]
    ) -> SubsetToll:
        """Both passes re-scored over the same rows, so the difference is the toll and nothing else."""
        return SubsetToll(
            label=label,
            studies=int(mask.sum()),
            kept=self.scores.mean(kept[mask], self.truth[mask]),
            foreign=self.scores.mean(foreign[mask], self.truth[mask]),
            interval=self.scores.mean_interval(foreign[mask], self.truth[mask]),
        )

    def read(self, name: str, kept: NDArray[np.float64], foreign: NDArray[np.float64]) -> list[SubsetToll]:
        """This member priced on the whole holdout and on each side of the partition, in that order.

        The pooled row is first and is not decoration: it is the number already in the ledger, so a run
        whose pooled toll does not reproduce is a run whose subsets may not be read at all.
        """
        whole = np.ones_like(self.coverage.fills)
        rows = [
            self.subset("pooled", kept, foreign, whole),
            self.subset("fills", kept, foreign, self.coverage.fills),
            self.subset("padded", kept, foreign, self.coverage.padded),
        ]
        for row in rows:
            logger.info("%s  %s", name, row.logged())
        return rows


def main() -> None:
    """Split an already-written per-study readout by whether the study could FILL the trained extent."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--per-study", type=Path, required=True, help="the .npz foreign_readout wrote")
    command.parser.add_argument("--metadata", type=Path, required=True, help="canonical_metadata.csv")
    command.parser.add_argument("--extent", type=float, default=FILL_MM, help="the field width the split asks about")
    arguments = command.parsed()

    dump = np.load(arguments.per_study)
    studies = [str(study) for study in dump["studies"]]
    truth = dump["truth"]
    metadata = pd.read_csv(arguments.metadata)

    coverage = Coverage.measured(studies, metadata, arguments.extent)
    logger.info(
        "%d studies fill %.0f mm, %d are padded up to it",
        int(coverage.fills.sum()),
        arguments.extent,
        int(coverage.padded.sum()),
    )

    split = PaddingSplit(LabelScores(), truth, coverage)
    lines: list[str] = []
    for name in split.members_of(dump.files):
        rows = split.read(name, dump[f"{name}{KEPT_SUFFIX}"], dump[f"{name}{FOREIGN_SUFFIX}"])
        lines.extend(f"{name}  {row.logged()}" for row in rows)

    command.report("\n".join(lines))


if __name__ == "__main__":
    main()
