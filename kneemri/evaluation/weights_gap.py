"""Where do THEIR weights beat OURS, per label, on studies both held out?

`python -m kneemri.evaluation.weights_gap --run band_dense_llrd_s2026`

THE QUESTION NO EXISTING RULER ANSWERS. `published_ruler` says in its own docstring that it "answers
questions about combining, not about whose weights are better", and that a comparison between ours and
theirs "still belongs on the shared fold-0 holdout" — and then nothing built it. So the project has two
per-label profiles that disagree about which label is cheap, measured on different studies with
different weights, and has already spent one arm (`focus_latmen`) on the wrong one of them:

  corpus ruler, THEIR weights, 4,407 studies:  Lateral Meniscus 0.7981 weakest, MCL 0.8166, Fracture 0.8923
  fold-0,       OUR weights,     909 studies:  MCL 0.7422 weakest, Fracture 0.7830, Lateral Meniscus 0.8423

Both can be right — they measure different models. This module removes the other two differences, so
what is left is the weights: same studies, same graded labels, same midpoint, same rank AUC.

WHY IT COSTS NOTHING. Their `oof` blocks and our `.holdout.npz` are both already on disk. No card, no
inference, no labels beyond the ones every arm was scored against.

TWO COMPARISONS, NOT ONE, AND THE DIFFERENCE BETWEEN THEM IS THE POINT. Their pool is four seeds; our
arm is a single member. Reporting only the pooled number would price ensembling as if it were weight
quality — the exact conflation the ledger warns about elsewhere. So both are reported: the mean of the
four seeds scored INDIVIDUALLY (single member against single member, the matched comparison) and their
seed-average (what an ensemble of theirs reaches). Ours beating their per-seed mean and losing to their
pool is a statement about ENSEMBLING; losing to both is a statement about WEIGHTS.

A FOLD THAT HELD OUT EVERYTHING IS NOT A FOLD. Their `fold: -1` members trained on the whole corpus and
still carry an `oof` block, as `oof_oracle` records. Scored here they would be reading their own training
data and would beat us everywhere for a reason that is not weight quality. Any fold covering more than
`HELD_OUT_SHARE` of the corpus is refused, loudly.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from core.findings import TARGETS
from core.paths import DataRoot
from kneemri.evaluation.published_ruler import PublishedFold
from kneemri.evaluation.scores import LabelScores
from kneemri.runs.holdout_record import HoldoutRecord

logger = logging.getLogger(__name__)

HELD_OUT_SHARE = 0.4
"""The largest share of the corpus one fold may hold out before it is not a held-out fold at all.

A five-fold split gives 0.2. A `fold: -1` member gives 1.0. Anything between is a split this reader has
never seen, and the honest response to it is a refusal rather than a number nobody can price."""


@dataclass(frozen=True)
class WeightsGap:
    """One label's AUC under our weights and under theirs, on the studies both held out."""

    label: str
    ours: float
    their_member: float
    their_pool: float
    positives: int

    def against_member(self) -> float:
        """Ours minus their average single member — the matched, ensembling-free comparison."""
        return self.ours - self.their_member

    def against_pool(self) -> float:
        """Ours minus their four-seed average, which is what their leaderboard entry actually runs."""
        return self.ours - self.their_pool


class MatchedRuler:
    """Our run and their published members, restricted to the studies both of them held out."""

    def __init__(self, published: Path) -> None:
        self.published = published

    def folds(self) -> list[PublishedFold]:
        """Every published member whose `oof` block is genuinely held out.

        The share is computed against the union of ALL recorded identifiers rather than a constant 4,407,
        so a partial download is refused for being partial instead of silently redefining the corpus.
        """
        entries = [PublishedFold.of(checkpoint) for checkpoint in sorted(self.published.glob("*.pt"))]
        if not entries:
            message = f"no published checkpoints under {self.published}"
            raise ValueError(message)
        corpus = len({name for entry in entries for name in entry.identifiers})
        kept = [entry for entry in entries if len(entry.identifiers) <= HELD_OUT_SHARE * corpus]
        for entry in entries:
            if entry not in kept:
                logger.info(
                    "REFUSED a member holding out %d of %d studies — trained on its own `oof`",
                    len(entry.identifiers),
                    corpus,
                )
        return kept

    def theirs(self, studies: list[str]) -> list[NDArray[np.float64]]:
        """One array per SEED, rows aligned to `studies` — the stacked ruler, restricted to our holdout.

        NO SINGLE MEMBER COVERS OUR FOLD, and that is not a defect in either split. Their five folds and
        our one were drawn independently, so any one of their members held out about a fifth of our 909
        studies. What covers the corpus is the STACK: within one seed the five folds partition all 4,407
        studies, so every study has exactly one member of that seed which did not train on it. This
        assembles that column and then takes our rows out of it.

        The seed is what makes a column a single MEMBER rather than an ensemble: four seeds means four
        independent readings of the same study, each by one model, which is the comparison our own
        single-member arm belongs in.
        """
        kept = self.folds()
        by_seed: dict[int, dict[str, NDArray[np.float64]]] = {}
        for entry in kept:
            rows = by_seed.setdefault(entry.seed, {})
            for position, name in enumerate(entry.identifiers):
                rows.setdefault(name, entry.values[position])
        columns = []
        for seed in sorted(by_seed):
            rows = by_seed[seed]
            missing = [name for name in studies if name not in rows]
            if missing:
                logger.info("seed %d is short %d of our studies — column dropped", seed, len(missing))
                continue
            columns.append(np.stack([rows[name] for name in studies]))
        return columns

    def compared(self, record: Path) -> list[WeightsGap]:
        """Per-label AUC for our run and for theirs, over our run's holdout studies."""
        reading = HoldoutRecord(record).read()
        if reading is None:
            message = f"no holdout predictions beside {record}"
            raise ValueError(message)
        members = self.theirs(reading.studies)
        if not members:
            message = f"no published seed covers all {len(reading.studies)} of this run's studies"
            raise ValueError(message)
        logger.info("%d published seed columns cover all %d studies", len(members), len(reading.studies))

        scores = LabelScores()
        ours = scores.per_label(reading.predictions, reading.truth)
        each = [scores.per_label(member, reading.truth) for member in members]
        pooled = scores.per_label(np.mean(members, axis=0), reading.truth)
        binary = reading.truth >= 0.5  # noqa: PLR2004 - LabelScores.MIDPOINT, quoted for the count only
        return [
            WeightsGap(
                label=label,
                ours=ours[label],
                their_member=float(np.mean([one[label] for one in each])),
                their_pool=pooled[label],
                positives=int(binary[:, index].sum()),
            )
            for index, label in enumerate(TARGETS)
        ]


def main() -> None:
    """Print the per-label gap between our weights and the published ones, worst deficit first."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="a run name or any of its own file names")
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    processed = DataRoot.from_config(arguments.paths).processed()
    record = Path(arguments.run) if Path(arguments.run).exists() else processed / f"{arguments.run}.pt"
    rows = MatchedRuler(processed / "published").compared(record)

    def line(label: str, cells: list[float], positives: str) -> str:
        return f"{label:<18}" + "".join(f"{cell:>9.4f}" for cell in cells) + f"{positives:>8}"

    logger.info("%-18s%9s%9s%9s%9s%9s%8s", "label", "ours", "their_1", "their_4", "vs_1", "vs_4", "n_pos")
    for row in sorted(rows, key=WeightsGap.against_member):
        cells = [row.ours, row.their_member, row.their_pool, row.against_member(), row.against_pool()]
        logger.info(line(row.label, cells, str(row.positives)))
    columns = [
        [row.ours for row in rows],
        [row.their_member for row in rows],
        [row.their_pool for row in rows],
        [row.against_member() for row in rows],
        [row.against_pool() for row in rows],
    ]
    logger.info(line("MEAN", [float(np.nanmean(column)) for column in columns], ""))


if __name__ == "__main__":
    main()
