"""Does the ensemble already use ALL of the acquisition-protocol signal, or only part of it?

WHERE THIS COMES FROM. `interpretations/kneemri/2026-08-24_slot_availability_is_signal.md` refuted the
suspicion that a study's slot-presence pattern injects SPURIOUS ordering: normalising ranks within a
presence stratum costs 0.0091 (by count) to 0.0209 (by pattern) mean AUC, negative on all twelve
labels. The between-stratum ordering is real information — which sequences a knee was scanned with
records what the ordering clinician suspected.

That refutation leaves one live question, and this module is it. Slot count ALONE reads mean
|AUC - 0.5| = 0.038 against the twelve labels. If the members had already extracted every bit of
that, an explicit per-pattern prior would add nothing on top of them. If they extracted only part —
plausible, since nothing in the loss ever names the presence pattern — then the remainder is free
AUC available with NO training and NO card, which is the rarest kind of gain this project has left.

WHY THE PRIOR IS FIT LEAVE-ONE-OUT. A per-pattern prevalence fit on the same 909 studies it is
scored on would read its own answer back. Each study's prior therefore excludes that study's own
label. LOO does not merely remove the leak, it OVERSHOOTS: dropping a positive lowers its own
pattern's prevalence, so the estimate is biased against the hypothesis. A gain measured this way is
a floor, which is the correct direction for a keep/kill.

WHAT IT WOULD MEAN AT TEST TIME. The prior is a table of twelve numbers per presence pattern, fit on
training studies and applied by reading the mask the pipeline already computes. No weights, no
encoder, no second pass over pixels.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float, Integer

from core import cli
from kneemri.runs.holdout_record import HoldoutRecord

logger = logging.getLogger(__name__)

CHAMPION = (
    "band_dense_long_s2026",
    "calib_span_s2026",
    "grade_ladder_s2026",
    "base_dense_s2026",
    "lr4e-5_ws_s2026",
    "through_dense_s2026",
)
"""The six champion seats that kept a `.holdout.npz`. Their rank average reproduces 0.8693 against
the champion's recorded 0.8694, which is the check that licenses every number this module prints."""

POSITIVE = 0.5
"""`truth` in a holdout record is CONTINUOUS (graded LLM-read labels), so a positive must be selected
by a threshold rather than summed. Summing it produced AUCs of 1.0466 and 0.1582 on first attempt —
two disagreeing denominators in one formula."""

BLENDS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
"""Weight on the prior's rank. 0.0 is the unblended ensemble, printed so the table carries its own
control rather than a number quoted from elsewhere."""

TARGETS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)


class Ordering:
    """The two rank statistics this module needs, written against numpy alone — `scipy` is not in the
    environment, and a metric that only reads ORDER has no business pulling one in."""

    @staticmethod
    def ranks(values: Float[np.ndarray, " n"]) -> Float[np.ndarray, " n"]:
        """Ordinal ranks in [0, 1]. Ties are broken arbitrarily rather than averaged; the members'
        predictions are continuous floats, so exact ties are vanishing and a tie-aware rank would buy
        nothing for the cost of a second sort."""
        order = np.argsort(np.argsort(values)).astype(np.float64)
        return order / max(len(values) - 1, 1)

    @staticmethod
    def auc(scores: Float[np.ndarray, " n"], labels: Float[np.ndarray, " n"]) -> float:
        """Mann-Whitney AUC over a CONTINUOUS truth column, binarised first. Returns `nan` when a
        label is constant on this split — a degenerate AUC is not 0.5, and averaging it in as 0.5
        would flatter every mean this module prints."""
        positive = labels > POSITIVE
        count = int(positive.sum())
        if count in (0, len(labels)):
            return float("nan")
        rank = Ordering.ranks(scores) * (len(scores) - 1) + 1.0
        return float((rank[positive].sum() - count * (count + 1) / 2) / (count * (len(labels) - count)))


@dataclass(frozen=True)
class Holdout:
    """The stored fold-0 predictions of one ensemble, aligned to a presence pattern per study."""

    studies: list[str]
    predictions: Float[np.ndarray, "n twelve"]
    truth: Float[np.ndarray, "n twelve"]
    presence: Integer[np.ndarray, "n slots"]

    @classmethod
    def load(cls, models: Path, processed: Path, names: tuple[str, ...]) -> "Holdout":
        """Rank-average `names`, then attach each study's slot presence from the cache mask.

        Members and mask are both joined by UID: the members to the first one's studies, the mask to its
        own index file. A member whose record is missing is a hard failure: silently averaging five
        seats and calling it the champion is the class of mistake this project keeps paying for.
        """
        first, *rest = (HoldoutRecord(models / f"{name}.pt").required() for name in names)
        readings = [first, *(reading.at(first.studies) for reading in rest)]
        stacked = np.stack(
            [
                np.stack([Ordering.ranks(reading.predictions[:, j]) for j in range(len(TARGETS))], axis=1)
                for reading in readings
            ]
        )
        mask = np.load(processed / "mask_336px_20sl_0.1-0.9band.npy")
        index = json.loads((processed / "index_336px_20sl_0.1-0.9band.json").read_text())
        where = {uid: position for position, uid in enumerate(index)}
        presence = np.stack([(mask[where[uid]] > 0).astype(np.int64) for uid in first.studies])
        return cls(first.studies, stacked.mean(axis=0), first.truth.astype(np.float64), presence)

    def patterns(self) -> Integer[np.ndarray, " n"]:
        """A stratum id per study: the presence bits read as a binary number."""
        bits = np.power(2, np.arange(self.presence.shape[1], dtype=np.int64))
        return (self.presence * bits).sum(axis=1)

    def report(self) -> dict[str, list[float]]:
        """AUC per label at each blend weight, plus the prior's own AUC standing alone."""
        prior = ProtocolPrior(self.patterns())
        table: dict[str, list[float]] = {}
        for j, target in enumerate(TARGETS):
            labels = self.truth[:, j]
            alone = prior.fit(labels)
            base = Ordering.ranks(self.predictions[:, j])
            blended = [Ordering.auc(base + weight * Ordering.ranks(alone), labels) for weight in BLENDS]
            table[target] = [*blended, Ordering.auc(alone, labels)]
        return table


@dataclass(frozen=True)
class ProtocolPrior:
    """A per-stratum per-label prevalence, fit leave-one-out so a study never reads its own label."""

    strata: Integer[np.ndarray, " n"]

    def fit(self, labels: Float[np.ndarray, " n"]) -> Float[np.ndarray, " n"]:
        """The LOO prevalence of one label within each study's own stratum.

        A stratum of size one has no leave-one-out estimate at all; it takes the global rate, which
        is the same as saying its pattern tells us nothing — correct, since with one member it
        cannot.
        """
        positive = (labels > POSITIVE).astype(np.float64)
        totals = np.zeros(len(positive))
        counts = np.zeros(len(positive))
        for stratum in np.unique(self.strata):
            here = self.strata == stratum
            totals[here] = positive[here].sum()
            counts[here] = here.sum()
        alone = counts <= 1
        estimate = np.where(alone, positive.mean(), (totals - positive) / np.maximum(counts - 1, 1))
        return estimate


def main() -> None:
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--members", nargs="*", default=list(CHAMPION))
    arguments = command.parsed()

    root = command.rooted(arguments)
    holdout = Holdout.load(root.models(), root.processed(), tuple(arguments.members))
    table = holdout.report()

    header = "  ".join(f"w={weight:<5}" for weight in BLENDS)
    logger.info(f"{'label':<18}{header}  prior-alone")
    for target, row in table.items():
        logger.info(f"{target:<18}" + "  ".join(f"{value:<7.4f}" for value in row))
    means = np.nanmean(np.array(list(table.values())), axis=0)
    logger.info(f"{'MEAN':<18}" + "  ".join(f"{value:<7.4f}" for value in means))
    logger.info(f"best blend {BLENDS[int(np.argmax(means[: len(BLENDS)]))]} at {means[: len(BLENDS)].max():.4f}")


if __name__ == "__main__":
    main()
