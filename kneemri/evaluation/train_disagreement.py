"""Which TRAINING studies the family cannot agree on: `python -m kneemri.evaluation.train_disagreement`.

`residual_split` measured, on the 909-study holdout, that ~70% of the pool's errors sit in a band more
than half but not all of the members get wrong, and that only ~5.4% are recoverable by any combiner. That
band is the largest addressable block this project has, and nothing in the training loop is aimed at it:
every arm on the ledger draws its epoch as a uniform permutation of the training positions.

WHAT THIS PRODUCES. Per training study, the spread of the members' predictions — the mean across the
twelve findings of the standard deviation across members. It is written beside the checkpoints as a
sampling weight so a trainer can draw hard-and-disagreed studies more often than easy-and-agreed ones,
and it is reported banded so the distribution can be read before anything is trained on it.

WHY SPREAD AND NOT LOSS. Hard-example mining conventionally ranks by the model's own loss, and on this
corpus that is the wrong signal: `uncertainty_split` puts ~87% of the residual's mass on the aleatoric
side, so the highest-loss studies are disproportionately the mislabelled ones and mining them harder
amplifies label noise. Spread across separately-trained members is an EPISTEMIC quantity — it is high
where the family is undecided and low where it agrees, including where it agrees on being wrong. That is
the middle band by construction, and it is the one quantity that distinguishes it from the unanimous
24.3% no sampler can help.

THE DENOMINATOR, AND THE LIMITATION THAT COMES WITH IT. These members all trained on these studies. The
spread is therefore IN-SAMPLE and is smaller than the same members would show on studies they had not
seen; a study every member memorised reads as agreement whether or not the family understands it. What
survives that bias is the ORDERING — a study the family still splits on after fitting it is genuinely
one it could not resolve — and the ordering is all a sampler consumes. It is not a held-out measurement
and nothing here may be quoted as one.

WHAT IT CANNOT SEE. It has no labels in it at all: a study where every member is confidently and
identically wrong scores zero spread and will be sampled LESS. That is deliberate — those are the
unanimous cells — but it means this weight is not a difficulty ranking, it is a disagreement ranking,
and the two differ exactly on the block a sampler cannot fix anyway.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.data.corpus.folds import ReportFolds
from core.denominator import Denominator
from core.wakelock import Wakelock
from kneemri.device import VisibleCards
from kneemri.inference.holdout_predictions import HoldoutPredictions
from kneemri.inference.member_readings import MemberReadings
from kneemri.inference.members import Member
from kneemri.inference.prediction_cache import PredictionCache

logger = logging.getLogger(__name__)

SPREAD_BANDS: tuple[tuple[float, float, str], ...] = (
    (0.00, 0.05, "agreed (spread < 0.05)"),
    (0.05, 0.10, "0.05 - 0.10"),
    (0.10, 0.15, "0.10 - 0.15"),
    (0.15, 1.00, "split (spread >= 0.15)"),
)
"""Spread strata. The edges are fixed rather than quantiles so two runs over different member sets are
comparable, and so a band that empties says the pool agreed rather than that the cut moved."""


@dataclass(frozen=True)
class SpreadRow:
    """One spread stratum: how many training studies fell in it and what weight they would draw at."""

    band: str
    studies: int
    mean_spread: float
    mean_weight: float

    def line(self) -> str:
        """This stratum's row."""
        return (
            f"  {self.band:<24} studies {self.studies:<6} mean spread {self.mean_spread:.4f}  "
            f"mean weight {self.mean_weight:.3f}"
        )


class TrainDisagreement:
    """The members' per-study disagreement over the studies they were all trained on."""

    def __init__(self, predictions: list[Float[NDArray[np.float64], "studies targets"]]) -> None:
        self.predictions = predictions

    def members(self) -> int:
        """How many members the spread is taken across."""
        return len(self.predictions)

    def spread(self) -> Float[NDArray[np.float64], " studies"]:
        """Per study, the mean across findings of the across-member standard deviation.

        The standard deviation is taken over the members' PROBABILITIES rather than their ranks, because
        a rank spread would be invariant to how confidently the family disagrees — and confidence is the
        whole difference between a study two members place either side of a threshold and one they place
        at 0.05 and 0.95.
        """
        stacked = np.stack(self.predictions, axis=0)
        return np.asarray(stacked.std(axis=0).mean(axis=1), dtype=np.float64)

    def weights(self, alpha: float) -> Float[NDArray[np.float64], " studies"]:
        """Sampling weights: a uniform floor mixed with the spread, normalised to mean 1.

        `alpha` is the share of the draw the spread controls, so 0.0 is exactly uniform and the arm that
        uses this can be its own control by setting it. The uniform floor is not a safety margin, it is
        what keeps every study in the epoch: a pure spread weight sends a study the family agrees on to
        near-zero probability, and the agreement of twelve in-sample members is not evidence enough to
        stop training on a study altogether.
        """
        spread = self.spread()
        scaled = spread / spread.mean() if spread.mean() > 0.0 else np.ones_like(spread)
        return np.asarray((1.0 - alpha) + alpha * scaled, dtype=np.float64)

    def banded(self, alpha: float) -> list[SpreadRow]:
        """One row per spread stratum."""
        spread = self.spread()
        weights = self.weights(alpha)
        rows: list[SpreadRow] = []
        for low, high, name in SPREAD_BANDS:
            inside = (spread >= low) & (spread < high)
            count = int(inside.sum())
            rows.append(
                SpreadRow(
                    band=name,
                    studies=count,
                    mean_spread=float(spread[inside].mean()) if count else 0.0,
                    mean_weight=float(weights[inside].mean()) if count else 0.0,
                )
            )
        return rows

    def report(self, alpha: float) -> list[str]:
        """The distribution of disagreement over the training split."""
        spread = self.spread()
        return [
            f"{self.members()} members over {len(spread)} training studies, IN-SAMPLE for all of them",
            f"spread: mean {spread.mean():.4f}  median {float(np.median(spread)):.4f}  "
            f"p90 {float(np.quantile(spread, 0.9)):.4f}  max {spread.max():.4f}",
            f"weights at alpha {alpha:.2f}: min {self.weights(alpha).min():.3f}  max {self.weights(alpha).max():.3f}",
            "by spread band",
            *[row.line() for row in self.banded(alpha)],
        ]


def main() -> None:
    """Score these members over the studies they trained on and write the disagreement weights."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--checkpoints", type=Path, nargs="+", required=True, help="members sharing ONE split")
    command.parser.add_argument("--fold", type=int, default=0)
    command.parser.add_argument("--folds", type=int, default=5)
    command.parser.add_argument("--split-seed", type=int, default=2026)
    command.parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="the share of the draw the spread controls in the REPORTED weights; the written file "
        "carries the raw spread, so a trainer picks its own alpha without rescoring",
    )
    command.parser.add_argument("--out", type=Path, required=True, help="where the per-study spread is written")
    command.parser.add_argument("--device", default=VisibleCards().device().type, choices=("cuda", "cpu"))
    arguments = command.parsed()

    root = command.rooted(arguments)
    predictions = HoldoutPredictions(torch.device(arguments.device))

    # THE SAME REFUSAL `member_curve` MAKES, for the opposite reason. There it keeps members that held
    # this fold out because scoring a member on studies it trained on reads high. Here every member is
    # scored on studies it trained on BY DESIGN, so the guard does not protect the number — it protects
    # the SPLIT: a member that held out a different fold trained on a different set, and its spread would
    # be a mix of in-sample and out-of-sample studies that no ordering can be read off.
    readings = MemberReadings(Member.held_out_from(arguments.fold, arguments.checkpoints), root.processed())
    corpus, reference = readings.reference()
    folds = ReportFolds(arguments.folds, arguments.split_seed)
    training, _ = folds.holdout(folds.assign(str(root.train_table()), reference), arguments.fold)
    Denominator(len(training), "this fold has no training studies to score").refuse_empty()

    cache = PredictionCache(
        root.processed() / "train_predictions",
        [reference[i] for i in training],
        predictions.views,
    )
    per_member: list[Float[NDArray[np.float64], "studies targets"]] = []
    with Wakelock():
        for member, slab, held in readings.slabs(training, reference, corpus):
            window = member.window(held.slices, overlap=False)
            per_member.append(predictions.cached(member, slab, cache, window))
            logger.info("%s read over %d training studies", member.file.name, len(training))

    disagreement = TrainDisagreement(per_member)
    np.savez(
        arguments.out,
        studies=np.array([reference[i] for i in training]),
        spread=disagreement.spread(),
    )
    logger.info("wrote %s", arguments.out)
    command.report(disagreement.report(arguments.alpha))


if __name__ == "__main__":
    main()
