"""What this card has already trained: `python -m kneemri.experiments.geometry_check`.

WHY IT EXISTS, measured before it was written. Over the 277 runs on disk carrying per-epoch timing, this
project has spent **307.6 hours of card**. Fingerprinting each run on its DATA AND GEOMETRY alone — the
fields that decide what pixels and what labels the trunk sees, and nothing about the head, the loss or
the optimiser — those 268 runs collapse to **128 distinct fingerprints**. 140 of them retrained a
geometry an earlier run had already trained, at a cost of **161.1 hours, 52% of every card-hour this
project has ever spent**. The largest single cluster is 41 runs and 53.6 hours on ONE fingerprint.

NOTHING IN THE HARNESS EVER ASKED THIS. `arm_check` reads an arm's flags, `card_check` reads the device,
`ledger_check` and `convergence_check` read afterwards. None of them can answer "has this geometry been
trained before?", so the question was never put, and the answer accumulated to half the card.

WHAT A REPEAT IS AND IS NOT. A repeat fingerprint is not waste by itself. Two runs sharing a geometry and
differing in head, loss or optimiser see identical pixels, so the second one COULD have warm-started from
the first at a fraction of the epochs — that is the recoverable part, roughly 100 of the 307 hours if
those short arms carry their own schedule. But a run that moved crop, span, resolution, slice count or
corpus is a real retrain: a checkpoint fitted at 130 mm is a bad initialisation at 160 mm, and warm-
starting it would measure the first geometry's optimum wearing the second one's name.

AND WHETHER THE REPEATS BOUGHT DIVERSITY IS A SEPARATE QUESTION THIS CHECK CANNOT ANSWER. The owner's
standing position is that a member which beats nothing is still worth having if it decorrelates the
ensemble. If those 41 head-and-loss variants are genuinely decorrelated then the 53.6 hours were the
cheapest diversity this project ever bought, and a warm start from a shared trunk is exactly what would
collapse them onto each other. M267 asks the researcher for that distribution. Until it returns, this
check REPORTS and never blocks — it tells a reader what already exists, and refuses nothing.

WHAT IT DELIBERATELY IGNORES. Seed and fold are part of the fingerprint, so two seeds of one recipe read
as different geometries here. That is not a mistake: a seed repeat is a legitimate noise measurement and
the project bans it for other reasons. The fields it ignores — head, loss, optimiser, rate, schedule,
EMA, mixup, patience, LoRA rank — are exactly the ones a warm start would let you vary cheaply.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from core import cli
from core.denominator import Denominator
from kneemri.runs.recorded_run import RecordedRun
from kneemri.runs.run_metrics import RunMetrics

logger = logging.getLogger(__name__)

GEOMETRY_KEYS = (
    "variant",
    "image_size",
    "slices",
    "crop_mm",
    "scheme",
    "centre",
    "band_low",
    "band_high",
    "spacing",
    "tie",
    "slot_pooling",
    "slice_pooling",
    "through_plane",
    "encoder_layer",
    "whole_stack",
    "labels",
    "studies",
    "train_fraction",
    "fold",
    "split_seed",
    "seed",
)
"""The recipe fields that decide WHAT THE TRUNK SEES, and no others.

Each one changes the pixels, the labels or the split. Everything absent from this tuple — head, loss,
rate, schedule, EMA, mixup, patience, LoRA rank, precision — leaves the input distribution alone, which
is precisely the condition under which a warm start from an earlier run is a valid initialisation rather
than a different experiment wearing the same name.

`seed` and `fold` ARE here, so two seeds of one recipe read as distinct. A seed repeat is a noise
measurement, not a redundant fit, and folding them together would have this check recommend warm-starting
the one thing a seed arm exists to vary.
"""

SECONDS = "seconds"
"""The row key holding an epoch's wall-clock. Summed rather than taken from a start-to-end difference,
because a run that sat waiting for the card between epochs would price that wait as training."""

ALONE = 1
"""Cluster size at which there is nothing to say. A fingerprint trained once is the ordinary case."""


@dataclass(frozen=True)
class TrainedGeometry:
    """One run reduced to what its trunk actually saw, plus the card time it cost."""

    name: str
    fingerprint: tuple[str, ...]
    hours: float

    @classmethod
    def of(cls, run: RunMetrics) -> "TrainedGeometry | None":
        """This run's geometry, or None when it recorded no recipe to read one from.

        A run with no recipe is not reported as a distinct geometry: an empty fingerprint would gather
        every pre-schema run into one enormous fake cluster and read as the finding of the century.
        """
        recipe = RecordedRun.of(run).recipe
        if recipe is None or not recipe.get("variant"):
            return None
        rows = [row for row in run.rows() if row.get(SECONDS) is not None]
        return cls(
            name=run.name(),
            fingerprint=tuple(str(recipe.get(key)) for key in GEOMETRY_KEYS),
            hours=sum(float(row[SECONDS]) for row in rows) / 3600,
        )


@dataclass(frozen=True)
class GeometryCluster:
    """Every run that trained one fingerprint, in the order they were found."""

    runs: tuple[TrainedGeometry, ...]

    @property
    def hours(self) -> float:
        """Card time the whole cluster cost."""
        return sum(run.hours for run in self.runs)

    @property
    def repeat_hours(self) -> float:
        """Card time spent on this geometry AFTER the first run had already trained it.

        The first run is never a repeat, however large the cluster: something had to fit the geometry.
        """
        return sum(run.hours for run in self.runs[1:])

    def line(self) -> str:
        """The cluster as one line — the count and the hours first, the names after."""
        names = ", ".join(run.name for run in self.runs)
        return f"{len(self.runs):3d} runs  {self.hours:6.1f} h  ({self.repeat_hours:5.1f} h repeat)  {names}"


class GeometryCheck:
    """Every run on disk, grouped by what its trunk saw rather than by what it was called."""

    def clusters(self, models: Path) -> list[GeometryCluster]:
        """Fingerprints trained more than once, costliest first.

        Grouped in DISCOVERY order inside each cluster so `repeat_hours` charges the later runs, and
        sorted alphabetically across clusters first so the ordering is a function of the disk rather
        than of the filesystem's iteration order.
        """
        grouped: dict[tuple[str, ...], list[TrainedGeometry]] = defaultdict(list)
        for metrics in sorted(models.glob("*.metrics.jsonl")):
            run = RunMetrics(metrics)
            if run.rows() and (seen := TrainedGeometry.of(run)) is not None:
                grouped[seen.fingerprint].append(seen)
        repeated = [GeometryCluster(tuple(runs)) for runs in grouped.values() if len(runs) > ALONE]
        return sorted(repeated, key=lambda cluster: -cluster.hours)

    def twins(self, models: Path, run: RunMetrics) -> list[TrainedGeometry]:
        """The runs that already trained THIS run's geometry — the question before queuing an arm.

        Excludes the run itself by name, so asking about a run already on disk returns the others rather
        than an answer containing its own subject.
        """
        asked = TrainedGeometry.of(run)
        if asked is None:
            return []
        return [
            seen
            for cluster in self.clusters(models)
            for seen in cluster.runs
            if seen.fingerprint == asked.fingerprint and seen.name != asked.name
        ]


def main() -> None:
    """Report every geometry this card has trained more than once, and what the repeats cost."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--run", type=Path, help="one metrics file: report only ITS twins")
    arguments = command.parsed()

    check = GeometryCheck()
    models = command.rooted(arguments).models()

    if arguments.run is not None:
        # A PATH, not a name, for `convergence_check`'s reason: a name resolves against the working
        # directory, finds nothing, and reports "no twins" — the sentence a genuinely novel geometry
        # produces, from a reading taken of no file at all.
        record = RunMetrics(arguments.run)
        if not record.exists():
            message = f"{record.path} is not on disk - --run takes a PATH to a metrics file, not a run name"
            raise SystemExit(message)
        found = check.twins(models, record)
        for seen in found:
            logger.info("already trained this geometry: %s (%.1f h)", seen.name, seen.hours)
        logger.info("%d run(s) share %s's geometry", len(found), record.name())
        return

    clusters = check.clusters(models)
    # The models root is gitignored and machine-local, so an unmounted drive reports "0 repeats" — the
    # sentence a perfectly disciplined project produces, from a check that read nothing at all.
    Denominator(len(clusters), f"no repeated geometry under {models}").refuse_empty()
    for cluster in clusters:
        logger.info("%s", cluster.line())
    logger.info(
        "%d repeated geometries, %d repeat runs, %.1f h of card spent refitting a trunk already fitted",
        len(clusters),
        sum(len(cluster.runs) - 1 for cluster in clusters),
        sum(cluster.repeat_hours for cluster in clusters),
    )
    logger.info("a repeat is a WARM-START CANDIDATE, not a verdict - see knee-364re and M267")


if __name__ == "__main__":
    main()
