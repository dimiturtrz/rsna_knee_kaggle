"""Score ensembles of members that are already on disk, with no card and no inference pass.

`python -m kneemri.evaluation.record_mix --records a.pt b.pt c.pt`

`member_curve` answers the same question and cannot run here: it rebuilds each member's reading from
the pixel cache, which needs the corpus, minutes, and — for anything not already in the prediction
cache — the GPU that a trainer is holding. But every run since `HoldoutRecord` existed WROTE the array
that analysis reconstructs: `Fitting` persists the kept epoch's per-study probabilities beside the
checkpoint, exact rather than re-derived. Forty-odd members are sitting there, and asking which of them
ensemble well has been costing inference passes over arrays that were already written.

WHAT IT IS FOR: diversity is the only ensemble lever this project has left (six axes measured, count
saturates at four, per-label selection is dead, the vote is flat), and diversity is a question about
WHICH members, which means a subset search over as many trained members as possible. That search is
arithmetic. It should not be gated on a card.

READ THE NUMBER AS A SIGN, NOT AS A SCORE. Two reasons, both recorded. A training-time record was
produced by the TRAINER's reading of the holdout, so a member fitted under a different window is
scored here as it scored itself and not as the submission path would read it — comparable across
members, not identical to `member_curve`'s GPU numbers. And the holdout prices diversity too cheaply
anyway: collapsing three members into one pixel group cost 0.0059 here and 0.014 on the board, 2.4x.
What transfers from an offline member comparison is the sign.

THE RULER IS CHECKED, not assumed. Records join by study NAME and their truth vectors must agree — this
project has three label tables and has already re-run its ablations after one of them moved, so a mix
across that boundary is refused rather than averaged over.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from kneemri.evaluation.member_curve import COMBINE_SEED, MemberCurve
from kneemri.evaluation.scores import LabelScores
from kneemri.runs.holdout_record import RULER_TOLERANCE, HoldoutReading, HoldoutRecord

logger = logging.getLogger(__name__)


class RecordMix:
    """The members' recorded holdout predictions, joined onto the studies all of them held out."""

    def __init__(self, records: list[Path]) -> None:
        self.records = records

    def readings(self) -> list[tuple[str, HoldoutReading]]:
        """Each record that exists, as (member name, its reading).

        A missing record is named and skipped rather than raised on: the pool predates the record file,
        so a sweep over every checkpoint in the processed directory would otherwise be decided by its
        oldest member. What must never be silent is a record that exists and disagrees, and that is
        `shared`'s refusal.
        """
        found = []
        for path in self.records:
            reading = HoldoutRecord(path).read()
            if reading is None:
                logger.info("%s: no holdout record, skipped", path.name)
                continue
            found.append((path.name, reading))
        return found

    def shared(self) -> list[str]:
        """The studies every member held out, in the first member's order.

        Ordered by one member rather than sorted, so the truth vector below is indexed the same way the
        first record's own numbers were — and an empty intersection is an error, not an empty table:
        members from different folds produce exactly that, and their mix would otherwise report a mean
        over nothing.
        """
        readings = self.readings()
        if not readings:
            message = "no member carries a holdout record"
            raise ValueError(message)
        common = set.intersection(*[set(reading.studies) for _, reading in readings])
        studies = [name for name in readings[0][1].studies if name in common]
        if not studies:
            message = f"no study is held out by all {len(readings)} members — different folds or splits"
            raise ValueError(message)
        return studies

    def per_member(self) -> "MixedRecords":
        """The members' names, their aligned predictions, and the one truth they are all scored against.

        Truth is checked against the first member's for every other member, which is the same refusal
        `HoldoutReading.aligned` makes pairwise: the arithmetic of a mix succeeds across two label
        tables and answers a question nobody asked.
        """
        studies = self.shared()
        readings = self.readings()
        names: list[str] = []
        predictions: list[Float[NDArray[np.float64], "shared labels"]] = []
        truth = self._rows(readings[0][1], studies)[1]
        for name, reading in readings:
            mine, theirs = self._rows(reading, studies)
            if not np.allclose(truth, theirs, atol=RULER_TOLERANCE, equal_nan=True):
                message = f"{name} was scored by a different label table over the {len(studies)} shared studies"
                raise ValueError(message)
            names.append(name)
            predictions.append(mine)
        return MixedRecords(names, predictions, truth)

    @staticmethod
    def _rows(
        reading: HoldoutReading, studies: list[str]
    ) -> tuple[Float[NDArray[np.float64], "shared labels"], Float[NDArray[np.float32], "shared labels"]]:
        """One reading's predictions and truth for the shared studies, in the shared order."""
        positions = {study: index for index, study in enumerate(reading.studies)}
        rows = np.array([positions[study] for study in studies], dtype=np.int64)
        return reading.predictions[rows], reading.truth[rows]


@dataclass(frozen=True)
class MixedRecords:
    """What several members predicted for the same studies, ready for `MemberCurve` to read.

    A type beside its producer rather than a bare tuple: the three arrays are only meaningful together —
    a name list reordered against a prediction list mislabels every row of the report and nothing fails.
    """

    names: list[str]
    predictions: list[Float[NDArray[np.float64], "shared labels"]]
    truth: Float[NDArray[np.float32], "shared labels"]

    def alone(self, scores: LabelScores) -> list[str]:
        """Each member's own mean score on the shared studies — the column every mix is read against."""
        return [
            f"{name} alone: {scores.mean(member, self.truth):.4f}"
            for name, member in zip(self.names, self.predictions, strict=True)
        ]


def main() -> None:
    """Score every subset of the given runs from their recorded holdout predictions, with no card."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True, help="runs sharing ONE split")
    parser.add_argument("--paired", action="store_true", help="per-label deltas against the FIRST run")
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="forward selection instead of every subset — the only tractable reading past a dozen members",
    )
    parser.add_argument("--combine-seed", type=int, default=COMBINE_SEED)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    scores = LabelScores()
    mixed = RecordMix(arguments.records).per_member()
    logger.info("%d members over %d shared studies", len(mixed.names), len(mixed.truth))
    for line in mixed.alone(scores):
        logger.info(line)
    curve = MemberCurve(scores, arguments.combine_seed)
    reading = (
        curve.greedy(mixed.predictions, mixed.truth, mixed.names)
        if arguments.greedy
        else curve.logged(mixed.predictions, mixed.truth, mixed.names, paired=arguments.paired)
    )
    for line in reading:
        logger.info(line)


if __name__ == "__main__":
    main()
