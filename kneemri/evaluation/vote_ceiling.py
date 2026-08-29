"""What is the VOTE worth, once the fitting is paid for?

`python -m kneemri.evaluation.vote_ceiling --roster comp/members.roster`

THE MEASUREMENT THE MODULE IT PRICES ASKED FOR AND NEVER GOT. `QualityWeightedRank` says in its own
docstring that it "has to be read on a split half, exactly as per-label selection was, and per-label
selection is what failed that test" — and nothing built that read. So the project ships `LogitAverage`
(`votefit_v1`, one tick over `dark6`, inside its own pre-registered flat band) and has never measured
what ANY weighting rule is worth after its weights are paid for. Every vote number in the ledger is a
fit-on-self number or a board tick inside a noise band.

WHY THE SPLIT HALF IS THE WHOLE INSTRUMENT. A weighted rule fitted on the studies it is then scored on
cannot lose: `LabelQualityRank` fits 12 numbers per member, and a rule with enough parameters reproduces
the holdout rather than the corpus. The honest question is what survives being fitted on studies the
score never sees, so weights come from one half and the score comes from the other, both ways round,
and the two halves are averaged. The FIT-ON-SELF number is reported beside it — not as a result, as the
ruler on the result: the gap between them is how much of the rule is fitting noise, and a rule whose
honest read is far under its own overfit read has been measured to overfit rather than assumed to.

WHY IT COSTS NOTHING. Holdout records are already on disk and no member is re-run. No card, no slot, no
inference — the same reason `weights_gap` was free.

WHAT IT CANNOT SAY. It reads the fold-0 holdout, so the transfer caveat is unchanged and a `_cf` or
`_ff` member is excluded by `RecordMix` rather than by judgement. And halving the studies roughly
doubles each per-label sampling error, so a rule separated from flat by less than the paired floor on a
HALF corpus is separated by nothing — the flat rule is scored on the identical halves precisely so the
comparison is paired and that inflation cancels where it can.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.denominator import Denominator
from core.paths import DataRoot
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.record_mix import RecordMix
from kneemri.evaluation.scores import LabelScores
from kneemri.inference.vote import (
    CHANCE,
    LabelQualityRank,
    LogitAverage,
    MedianLogit,
    QualityWeightedRank,
    RankAverage,
    Vote,
)

logger = logging.getLogger(__name__)

HALVES = 2
"""Two halves and not k folds. The question is whether a fitted weight survives unseen studies at all,
and the cheapest honest answer to that is one split read both ways; a finer split buys precision on a
number that only matters if it is large."""


@dataclass(frozen=True)
class RuleReading:
    """One vote rule's honest score and the overfit score that bounds it.

    Both numbers or neither: an honest read alone cannot say whether a rule is weak or merely
    over-parameterised, and an overfit read alone is the mistake this module exists to stop.
    """

    rule: str
    honest: float
    fitted_on_self: float
    against_flat: float

    def line(self) -> str:
        """The row a decision is read off — honest first, because that is the only one that transfers."""
        return (
            f"  {self.rule:<22} honest {self.honest:.4f}  "
            f"vs flat {self.against_flat:+.4f}  (fit-on-self {self.fitted_on_self:.4f}, "
            f"overfit gap {self.fitted_on_self - self.honest:+.4f})"
        )


@dataclass(frozen=True)
class VoteCeiling:
    """A pool read under every vote rule the project owns, with the fitting paid for."""

    pool: Pool

    @classmethod
    def over(cls, paths: list[Path]) -> "VoteCeiling":
        """Every named checkpoint that carries a mixable holdout record.

        A dropped member is logged rather than skipped silently: a vote rule compared over a pool
        shorter than the caller named is the wrong-denominator failure, and it would land as a
        statement about the RULE.
        """
        mix = RecordMix(paths)
        for dropped in mix.narrow():
            logger.warning("%s cannot be mixed with the rest, dropped from the reading", dropped)
        pool = mix.per_member()
        Denominator(len(pool), f"none of the {len(paths)} named checkpoints carries a holdout record").refuse_empty()
        logger.info("reading %d members over %d studies", len(pool), len(pool.truth))
        return cls(pool)

    def halves(self) -> list[NDArray[np.bool_]]:
        """Studies split in two by POSITION, which is a deterministic split and not a seed.

        Alternating rather than cutting the array in half: the holdout is built in corpus order and a
        contiguous cut would put any ordering structure the corpus carries — acquisition site, date,
        anything — entirely in one half, so the two halves would differ by more than sampling.
        """
        index = np.arange(len(self.pool.truth))
        return [(index % HALVES) == half for half in range(HALVES)]

    def quality(self, scores: LabelScores, on: NDArray[np.bool_]) -> Float[np.ndarray, "members targets"]:
        """Each member's per-label AUC over the named studies — the table every weighted rule fits on.

        An undefined label (one class present in this half) is read as CHANCE, which weights it to zero
        rather than propagating a NaN into every member's weight for that label.
        """
        table = [list(scores.per_label(member[on], self.pool.truth[on]).values()) for member in self.pool.predictions]
        return np.nan_to_num(np.asarray(table, dtype=np.float64), nan=CHANCE)

    def rules(self, fitted: Float[np.ndarray, "members targets"]) -> list[Vote]:
        """Every rule the project owns, the weighted ones carrying weights fitted on the OTHER half."""
        return [
            RankAverage(),
            QualityWeightedRank(list(fitted.mean(axis=1))),
            LabelQualityRank(fitted),
            LogitAverage(),
            MedianLogit(),
        ]

    def scored(self, scores: LabelScores, rule: Vote, on: NDArray[np.bool_]) -> float:
        """One rule's mean AUC over the named studies, combining every member in the pool."""
        combined = rule.combine([member[on] for member in self.pool.predictions])
        return float(np.nanmean(list(scores.per_label(combined, self.pool.truth[on]).values())))

    def read(self, scores: LabelScores) -> list[RuleReading]:
        """Every rule, fitted on each half and scored on the other, averaged over both directions."""
        first, second = self.halves()
        honest = [
            [self.scored(scores, rule, score_on) for rule in self.rules(self.quality(scores, fit_on))]
            for fit_on, score_on in ((first, second), (second, first))
        ]
        overfit = [
            [self.scored(scores, rule, score_on) for rule in self.rules(self.quality(scores, score_on))]
            for score_on in (second, first)
        ]
        names = [rule.named() for rule in self.rules(self.quality(scores, first))]
        means = np.asarray(honest).mean(axis=0), np.asarray(overfit).mean(axis=0)
        return [
            RuleReading(name, float(means[0][index]), float(means[1][index]), float(means[0][index] - means[0][0]))
            for index, name in enumerate(names)
        ]

    def lines(self, scores: LabelScores) -> list[str]:
        """The reading as a report, flat first because every other rule is a delta against it."""
        readings = self.read(scores)
        rows = [f"vote rules over {len(self.pool)} members, weights fitted on one half and scored on the other:"]
        rows.extend(reading.line() for reading in readings)
        best = max(readings, key=lambda reading: reading.honest)
        rows.append(f"BEST honest rule: {best.rule} at {best.honest:.4f} ({best.against_flat:+.4f} vs flat)")
        return rows


def main() -> None:
    """Report what each vote rule scores once its weights are fitted on studies it is not scored on."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--roster", type=Path, default=None, help="a roster file, one member per line")
    command.parser.add_argument(
        "--candidates", nargs="*", default=[], help="member names or checkpoint paths, instead of a roster"
    )
    arguments = command.parsed()

    root = DataRoot.from_config(arguments.paths)
    named = arguments.roster.read_text(encoding="utf-8").splitlines() if arguments.roster else arguments.candidates
    paths = root.checkpoints(list(named))
    Denominator(len(paths), "no checkpoint named by --roster or --candidates").refuse_empty()

    for line in VoteCeiling.over(paths).lines(LabelScores()):
        logger.info(line)


if __name__ == "__main__":
    main()
