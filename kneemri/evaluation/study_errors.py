"""Which HOLDOUT STUDIES does a run get wrong, and is the defect the model's or the label's?

`python -m kneemri.evaluation.study_errors --run <name>`

LEVEL 1 IS ALREADY ANSWERED AND IT CLOSED. `weights_gap` (knee-q1pf) compared our weights against twenty
published CC0 members on these same 909 studies: MCL 0.7423 is our weakest column and Fracture 0.7827 the
second, but we lead on ten of twelve labels, so the whole demonstrated per-label headroom is +0.0043 on
the mean — a quarter of the bar, and a CEILING rather than an estimate. A model that leads the field on
ten of twelve columns is not mis-balanced and is not mis-weighted, so neither "the loss" nor "the data
balance" survives as an account of what it gets wrong. The share question is spent.

WHAT IS NOT ANSWERED is which STUDIES carry the error, and that is a different unit with a different set
of available causes. Nothing here ranks studies: `competence_axis` is members x findings, `member_distance`
is pairwise and label-free, and `uncertainty_split` reads what an ensemble does not know without asking
whether it was right.

THE UNIT IS A DISCORDANT PAIR, NOT A RESIDUAL, because the competition metric is a ranking one. A study
predicted at 0.6 against a positive label is not wrong if every negative sits below 0.6, and |p - y| would
charge it 0.4 for being correctly ranked. So a study's error is the share of opposite-class studies it is
mis-ordered against, which is EXACT rather than a proxy: per label the AUC is one minus the discordant
fraction, every discordant pair has exactly one positive and one negative member, and summing the per-study
counts recovers twice the total. The decomposition is arithmetic, and it needs no threshold.

WHAT A TAIL MEANS, and the third cause neither of the two obvious ones can see. A study can score badly
because the model is weak on it, or because the LABEL IS WRONG: `reader_agreement` counts 108 confident
false positives against 32 misses over the 58 radiologist-annotated studies, and this holdout's truth is
report-derived exactly like the training labels. If the worst-ranked studies are the ones where the report
extractor and the radiologist disagree, then the per-label numbers above are measured against partly-bad
truth and the weak-column reading is a fact about the RULER.

THE DENOMINATOR THAT MATTERS, stated before any tail is read. Only 58 studies in the corpus carry a
radiologist annotation, and a fold-0 holdout is 909 of 4,407, so roughly a dozen gold studies land here.
That is the honest power of the label-disagreement test and it is reported as a count, never as a rate
over the tail: a rate computed inside the subset that happens to have gold is not a rate over the corpus.
Where the count is too small to separate anything, the row says so rather than quoting a difference.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.findings import TARGETS
from kneemri.evaluation.gold_rows import GoldRows
from kneemri.evaluation.scores import MIDPOINT
from kneemri.runs.holdout_record import HoldoutRecord

LOGGER = logging.getLogger(__name__)

TIE_CREDIT = 0.5
"""A tied pair is half discordant, which is the same convention the rank AUC itself uses."""

TAIL_FRACTION = 0.1
"""The share of studies read as the tail. A tenth of 909 is 91 studies — large enough that a covariate
difference has a denominator, small enough that it is a tail rather than a re-description of the mean."""

REPORTED_TAIL = 12
"""How many individual studies are named in the log. The ranking is written whole to CSV; this is the
part a reader looks at, and a longer list is scrolled past rather than read."""

GOLD_FLOOR = 30
"""How many gold studies the holdout needs before a tail-versus-body gold difference is quoted at all.

A tenth of the corpus is the tail, so G gold studies put 0.1G of them there in expectation and the
count's own Poisson spread is sqrt(0.1G). Separating even a DOUBLING of gold density from that spread
needs 0.1G above about 3, hence 30. A fold-0 holdout carries roughly a dozen, so this refusal is the
expected outcome and not an edge case: the threshold exists to stop the tail being read as evidence
about labels on a denominator that cannot carry it."""


@dataclass(frozen=True)
class StudyDamage:
    """One study's exact share of a run's ranking error, and the finding that carries it."""

    study: str
    damage: float
    positives: int
    worst_finding: str
    worst_damage: float
    over: float
    under: float

    @property
    def direction(self) -> str:
        """Whether this study loses its pairs by being ranked too HIGH or too LOW.

        The two are not symmetric defects. Ranked too high on a cell the label calls negative is the
        shape of the report extractor's dominant error — 108 confident false positives against 32
        misses over the annotated studies — so a tail made of over-ranked studies is a claim about the
        RULER that can be tested on all 909, rather than on the dozen carrying a radiologist.

        BOTH SIDES ARE PER-CELL MEANS, NOT SUMS. A study carries about 3.6 positive cells against 8.4
        negative ones, so a sum calls almost every study over-ranked on the class sizes alone and the
        tail's share would be a fact about label prevalence. The mean puts one negative cell and one
        positive cell on the same footing, which is the only version of this comparison that reads.
        """
        return "over" if self.over > self.under else "under"


class StudyErrors:
    """The per-study decomposition of a run's holdout AUC into the pairs each study loses."""

    @staticmethod
    def discordant(
        predictions: Float[NDArray[np.float64], " studies"],
        positive: NDArray[np.bool_],
    ) -> Float[NDArray[np.float64], " studies"]:
        """Each study's share of the opposite class it is mis-ordered against, for ONE finding.

        A positive is charged for every negative scored above it, a negative for every positive scored
        below it, and a tie costs both of them half. The result is a rate against the study's own
        opposite-class count, so a positive and a negative are on the same scale despite the classes
        being different sizes.
        """
        share = np.zeros_like(predictions)
        negative = ~positive
        if not positive.any() or not negative.any():
            return share
        gaps = predictions[:, None] - predictions[None, :]
        above = (gaps > 0).astype(np.float64) + TIE_CREDIT * (gaps == 0).astype(np.float64)
        share[positive] = above[np.ix_(negative, positive)].sum(axis=0) / int(negative.sum())
        share[negative] = above[np.ix_(negative, positive)].sum(axis=1) / int(positive.sum())
        return share

    def damages(
        self,
        predictions: Float[NDArray[np.float64], "studies findings"],
        truth: Float[NDArray[np.floating], "studies findings"],
    ) -> Float[NDArray[np.float64], "studies findings"]:
        """The discordant share of every (study, finding) cell, in the array's own order."""
        table = np.zeros(predictions.shape, dtype=np.float64)
        for column in range(predictions.shape[1]):
            positive = truth[:, column] > MIDPOINT
            table[:, column] = self.discordant(predictions[:, column].astype(np.float64), positive)
        return table

    @staticmethod
    def sided(
        row: Float[NDArray[np.float64], " findings"],
        mask: NDArray[np.bool_],
    ) -> float:
        """The mean discordant share over one study's cells of a single class, zero when it has none."""
        return float(row[mask].mean()) if bool(mask.any()) else 0.0

    @staticmethod
    def ranked(
        studies: list[str],
        table: Float[NDArray[np.float64], "studies findings"],
        truth: Float[NDArray[np.floating], "studies findings"],
    ) -> list[StudyDamage]:
        """Studies worst-first by mean discordant share over the twelve findings."""
        findings = list(TARGETS)
        mean = table.mean(axis=1)
        worst = table.argmax(axis=1)
        positive = truth > MIDPOINT
        rows = [
            StudyDamage(
                study=study,
                damage=float(mean[index]),
                positives=int(positive[index].sum()),
                worst_finding=findings[int(worst[index])],
                worst_damage=float(table[index, int(worst[index])]),
                over=StudyErrors.sided(table[index], ~positive[index]),
                under=StudyErrors.sided(table[index], positive[index]),
            )
            for index, study in enumerate(studies)
        ]
        return sorted(rows, key=lambda row: row.damage, reverse=True)


@dataclass(frozen=True)
class TailProfile:
    """What separates the worst studies from the rest, with the counts the separation rests on."""

    tail: int
    body: int
    tail_positives: float
    body_positives: float
    tail_gold: int
    body_gold: int
    tail_over: int
    body_over: int

    @classmethod
    def cut(cls, rows: list[StudyDamage], gold: set[str]) -> "TailProfile":
        """Split the ranking at the tail fraction and count what differs across the cut."""
        edge = max(1, int(len(rows) * TAIL_FRACTION))
        tail, body = rows[:edge], rows[edge:]
        return cls(
            tail=len(tail),
            body=len(body),
            tail_positives=float(np.mean([row.positives for row in tail])),
            body_positives=float(np.mean([row.positives for row in body])),
            tail_gold=sum(row.study in gold for row in tail),
            body_gold=sum(row.study in gold for row in body),
            tail_over=sum(row.direction == "over" for row in tail),
            body_over=sum(row.direction == "over" for row in body),
        )

    def report(self) -> list[str]:
        """The lines a reader is given, including the ones that refuse to quote a rate."""
        lines = [
            f"tail {self.tail} studies, body {self.body}",
            f"positives per study: tail {self.tail_positives:.2f} vs body {self.body_positives:.2f}",
            f"over-ranked (model says yes, label says no): tail {self.tail_over}/{self.tail}"
            f" vs body {self.body_over}/{self.body}",
            f"radiologist-annotated studies in this holdout: tail {self.tail_gold}, body {self.body_gold}",
        ]
        if self.tail_gold + self.body_gold < GOLD_FLOOR:
            lines.append(
                "TOO FEW GOLD STUDIES TO SEPARATE ANYTHING — the label-disagreement half of this question"
                " is not answerable on a fold-0 holdout, and a difference quoted over this count would be"
                " a rate inside the subset that happens to carry gold."
            )
        return lines


def main() -> None:
    """Rank a run's holdout studies by the pairs they lose, and profile the tail."""
    command = cli.Command(__doc__, LOGGER)
    command.parser.add_argument("--run", type=str, required=True, help="checkpoint stem the holdout was written beside")
    command.parser.add_argument("--out", type=Path, default=None, help="where the whole ranking is written")
    arguments = command.parsed()
    root = command.rooted(arguments)
    reading = HoldoutRecord(root.models() / f"{arguments.run}.pt").read()
    if reading is None:
        message = f"no holdout record beside {arguments.run}"
        raise SystemExit(message)
    errors = StudyErrors()
    table = errors.damages(reading.predictions, reading.truth)
    rows = errors.ranked(reading.studies, table, reading.truth)
    annotated, _ = GoldRows().annotated(root.train_table())
    gold = set(annotated)
    LOGGER.info(f"{arguments.run} epoch {reading.epoch}: {len(rows)} studies, mean damage {np.mean(table):.4f}")
    for row in rows[:REPORTED_TAIL]:
        LOGGER.info(
            f"  {row.study} damage {row.damage:.4f} positives {row.positives} {row.direction}-ranked"
            f" worst {row.worst_finding} {row.worst_damage:.4f}"
        )
    for line in TailProfile.cut(rows, gold).report():
        LOGGER.info(line)
    if arguments.out is not None:
        header = "study,damage,positives,worst_finding,worst_damage,over,under,direction\n"
        body = "".join(
            f"{row.study},{row.damage:.6f},{row.positives},{row.worst_finding},{row.worst_damage:.6f},"
            f"{row.over:.6f},{row.under:.6f},{row.direction}\n"
            for row in rows
        )
        arguments.out.write_text(header + body, encoding="utf-8")
        LOGGER.info(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
