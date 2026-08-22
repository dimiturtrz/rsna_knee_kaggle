"""Score ensembles of members that are already on disk, with no card and no inference pass.

`python -m kneemri.evaluation.record_mix --records a.pt b.pt c.pt`

`member_curve` answers the same question and cannot run here: it rebuilds each member's reading from
the pixel cache, which needs the corpus, minutes, and — for anything not already in the prediction
cache — the GPU that a trainer is holding. But every run since `HoldoutRecord` existed WROTE the array
that analysis reconstructs: `Fitting` persists the kept epoch's per-study probabilities beside the
checkpoint, exact rather than re-derived. Forty-odd members are sitting there, and asking which of them
ensemble well has been costing inference passes over arrays that were already written.

WHAT IT IS FOR: diversity is the only ensemble lever this project has left (six axes measured, count
plateaus at eight, per-label selection is dead, the vote is flat), and diversity is a question about
WHICH members, which means a subset search over as many trained members as possible. That search is
arithmetic. It should not be gated on a card.

COUNT PLATEAUS, IT DOES NOT SATURATE AT FOUR — this line said "four" until 2026-08-22, when the greedy
was finally run over all 70 same-split records rather than 22. Four reads 0.8703, five 0.8710, the peak
0.8717 at eight to ten, and twenty-two still 0.8711: five through twenty-two spans 0.0007, so four is
0.0014 BELOW a seventeen-member-wide plateau. The board agrees from its own side (best entry per count
0.930 / 0.931 / 0.931 / 0.932 for four through seven). Both earlier readings were true of the pool they
were taken on, which is the failure mode to watch for here: this tool's answer moves as the pool grows,
so quote the pool size with the number or the next reader inherits a stale constant.

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

from kneemri.evaluation.member_curve import COMBINE_SEED, MemberCurve, Pool
from kneemri.evaluation.scores import LabelScores
from kneemri.runs.holdout_record import RULER_TOLERANCE, HoldoutReading, HoldoutRecord

logger = logging.getLogger(__name__)

SHARED_FLOOR = 0.9
"""How much of the widest member's holdout the shared set must keep before a mix is believed.

Not a tolerance for disagreement — members on one split agree exactly, so the real values are 1.0 or a
fraction near zero. It is a floor with room for a member whose record predates a handful of studies.
"""


class RecordMix:
    """The members' recorded holdout predictions, joined onto the studies all of them held out."""

    def __init__(self, records: list[Path]) -> None:
        self.records = records
        self._readings: list[tuple[str, HoldoutReading]] | None = None

    def readings(self) -> list[tuple[str, HoldoutReading]]:
        """Each record that exists, as (member name, its reading).

        A missing record is named and skipped rather than raised on: the pool predates the record file,
        so a sweep over every checkpoint in the processed directory would otherwise be decided by its
        oldest member. What must never be silent is a record that exists and disagrees, and that is
        `shared`'s refusal.

        Read once and held: `shared` and `per_member` both want them, so an uncached read decodes every
        npz three times and prints each skipped member's line three times, which reads as three
        different missing records rather than one.
        """
        if self._readings is not None:
            return self._readings
        found = []
        for path in self.records:
            reading = HoldoutRecord(path).read()
            if reading is None:
                logger.info("%s: no holdout record, skipped", path.name)
                continue
            found.append((path.name, reading))
        self._readings = found
        return found

    def shared(self) -> list[str]:
        """The studies every member held out, in the first member's order.

        Ordered by one member rather than sorted, so the truth vector below is indexed the same way the
        first record's own numbers were — and an empty intersection is an error, not an empty table:
        members from different folds produce exactly that, and their mix would otherwise report a mean
        over nothing.

        A SMALL intersection is refused for the same reason an empty one is. A smoke run holds out
        eleven studies; sweeping the processed directory put four of them beside twenty-three real
        members, and the search silently reported a greedy ordering over those eleven — a number that
        looked like every other number this tool prints. The denominator has to be checked, not
        assumed, so a member that costs the shared set most of its studies is named and the mix is
        refused rather than quietly reported over the survivors.
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
        widest = max(len(reading.studies) for _, reading in readings)
        if len(studies) < SHARED_FLOOR * widest:
            narrow = sorted(name for name, reading in readings if len(reading.studies) < widest)
            message = (
                f"{len(studies)} shared studies against {widest} in the widest member — "
                f"{', '.join(narrow)} hold out a different split (smoke runs and other folds do this)"
            )
            raise ValueError(message)
        return studies

    def per_member(self) -> Pool:
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
        return Pool(names, predictions, truth)

    @staticmethod
    def _rows(
        reading: HoldoutReading, studies: list[str]
    ) -> tuple[Float[NDArray[np.float64], "shared labels"], Float[NDArray[np.float32], "shared labels"]]:
        """One reading's predictions and truth for the shared studies, in the shared order."""
        positions = {study: index for index, study in enumerate(reading.studies)}
        rows = np.array([positions[study] for study in studies], dtype=np.int64)
        return reading.predictions[rows], reading.truth[rows]


@dataclass(frozen=True)
class MixReading:
    """Which of the three questions an invocation is asking of one pool.

    Ordered by how much is held fixed. A roster pins the members that already ship and prices the rest
    against them; greedy pins nothing and builds an ensemble from scratch; the exhaustive reading pins
    nothing and enumerates. The roster case is checked first because naming one is unambiguous — nobody
    passes a shipped four-member roster and means `search every subset`.
    """

    pool: Pool
    roster: list[Path]

    def lines(self, curve: MemberCurve, *, greedy: bool, paired: bool, labels: bool) -> list[str]:
        """The reading this invocation asked for, as report lines."""
        if labels:
            # BEFORE the roster branch, because a label reading of the SHIPPED mix is the common case
            # and it is named with the same `--roster` the seat scan is named with. A roster plus
            # `--labels` means "open up the entry I ship", never "price candidates against it".
            held, _ = self.pool.split(self.roster) if self.roster else (self.pool, None)
            return curve.labels(held)
        if self.roster:
            held, candidates = self.pool.split(self.roster)
            # BOTH readings, because they answer different questions and only one of them is a decision:
            # the add says whether a candidate brings anything the roster lacks, the swap says whether to
            # ship it and at whose expense. Printing only the add is what let every seat number quoted in
            # this repository be the price of a fifth member the pin forbids.
            return curve.marginal(held, candidates) + curve.swap(held, candidates)
        if greedy:
            return curve.greedy(self.pool)
        return curve.logged(self.pool, paired=paired)


def main() -> None:
    """Score every subset of the given runs from their recorded holdout predictions, with no card."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True, help="runs sharing ONE split")
    parser.add_argument("--paired", action="store_true", help="per-label deltas against the FIRST run")
    parser.add_argument(
        "--roster",
        type=Path,
        nargs="+",
        default=(),
        help="hold these members fixed and report what each OTHER record would add — the seat number",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="forward selection instead of every subset — the only tractable reading past a dozen members",
    )
    parser.add_argument(
        "--labels",
        action="store_true",
        help="the mix's own per-label AUCs, worst first, each priced in units of the final score",
    )
    parser.add_argument("--combine-seed", type=int, default=COMBINE_SEED)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    scores = LabelScores()
    roster = list(arguments.roster)
    everything = list(dict.fromkeys([*arguments.records, *roster]))
    mixed = RecordMix(everything).per_member()
    logger.info("%d members over %d shared studies", len(mixed.names), len(mixed.truth))
    for line in mixed.alone(scores):
        logger.info(line)
    curve = MemberCurve(scores, arguments.combine_seed)
    reading = MixReading(mixed, roster)
    for line in reading.lines(curve, greedy=arguments.greedy, paired=arguments.paired, labels=arguments.labels):
        logger.info(line)


if __name__ == "__main__":
    main()
