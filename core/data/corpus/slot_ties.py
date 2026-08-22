"""How often two series compete for one slot — and whether that ambiguity is ever resolved differently.

`SlotScheme.assign` picks the densest stack for a slot and breaks a tie with `max`, which keeps the
FIRST maximal element in list order. The published members we grafted were routed by a pandas
`sort_values` that keeps the first in FRAME order. Where a slot is contested the two rules CAN pick
different series, and a different series in a slot is a different image under the head — nothing
downstream recovers it. Two independent measurements said the difference costs us something: our
pipeline reproduces those members' out-of-fold predictions only to max|diff| 0.025, and the graft
scored 0.888 against the public forks' ~0.899 on identical weights.

THE HYPOTHESIS IS REFUTED, and by the cheapest thing in this module rather than the most careful.
`disagreeing_order` reads one CSV: the frame order of `train_series.csv` is the sorted-UID order for
ALL 4,407 studies, and `Study.from_directory` probes `sorted(directory.iterdir())` whose names ARE
those UIDs. Same input order, and both rules keep the first maximal by slice count — so the two
pipelines make the SAME choice in every contested slot. The 1,100 rivalries are opportunity that is
never taken.

WHAT SURVIVES: the census. 20.3% of studies hold a contested slot and 96% of those hold genuinely
different pictures, so the routing IS load-bearing and a future change to it would move real images.
What does not survive is tie-break as the explanation for the 0.011-0.025 residual, which must live in
the other candidates — window policy, laterality, crop geometry, interpolation.

THE ORDER OF THE TWO MEASUREMENTS WAS BACKWARDS. Counting contested slots takes 4,407 DICOM header
reads and answers whether a divergence is POSSIBLE; comparing the two orderings takes one CSV read and
answers whether it HAPPENS. The second strictly dominates, and it was available on day one.

It reads one DICOM header per series and no pixels, so it is cheap and runs happily beside training.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core.data.corpus.slots import SLOTS, SlotScheme
from core.data.corpus.study import Study

PERCENT = 100.0


class SlotTies:
    """Counts contested slots across a corpus of studies."""

    def __init__(self, scheme: SlotScheme | None = None) -> None:
        self.scheme = scheme or SlotScheme()

    def disagreeing_order(self, series_table: Path) -> tuple[int, int]:
        """Studies whose CSV ROW ORDER differs from the SORTED-UID order we route by, and the total.

        THE MEASUREMENT THAT DECIDES WHETHER ANY OF THE REST MATTERS, and it costs a CSV read.

        A tie can only be resolved differently by two pipelines if the two pipelines see the candidates
        in a different ORDER — both rules keep the first maximal by slice count, so identical input
        order forces an identical choice no matter how many slots are contested. `Study.from_directory`
        probes `sorted(directory.iterdir())`, and the directory names ARE the series UIDs, so our list
        order is the sorted-UID order. Theirs is the frame order of this table.

        Counting contested slots (20.3% of studies, 1,100 rivalries, 96% of them genuinely different
        pictures) measures the OPPORTUNITY to diverge. This measures whether the opportunity is ever
        taken. It is the denominator the rest of this module does not have.
        """
        frame = pd.read_csv(series_table)
        grouped = frame.groupby("StudyInstanceUID", sort=False)["SeriesInstanceUID"].apply(list)
        return sum(1 for series in grouped if list(series) != sorted(series)), len(grouped)

    def of_study(self, directory: Path, planes: dict[str, str]) -> dict[str, int]:
        """Which of this study's slots were won by a tie on slice count."""
        study = Study.from_directory(directory, planes)
        return self.scheme.contested(study.acquisitions, planes)

    def lost_of_study(self, directory: Path, planes: dict[str, str]) -> Counter[str]:
        """What this study's unrouted series ARE, keyed by the predicate axes that failed to match.

        Counting them is not enough to decide anything: a discarded localiser is worth nothing and a
        discarded axial T1 may be worth a great deal. The key carries plane, weighting and fat
        suppression so the answer says WHICH sequences the six-slot scheme cannot express, and the
        slice total says how much imaging goes with them.
        """
        study = Study.from_directory(directory, planes)
        lost: Counter[str] = Counter()
        for acquisition in self.scheme.unrouted(study.acquisitions, planes):
            plane = planes.get(acquisition.series, "unknown")
            fluid = "fluid" if acquisition.fluid_sensitive else "t1"
            suppression = "fs" if acquisition.fat_suppressed else "nofs"
            lost[f"{plane}|{fluid}|{suppression}"] += 1
            lost["__slices__"] += acquisition.slice_count
        lost["__series__"] += len(study.acquisitions)
        return lost

    @staticmethod
    def _lost_of(argument: tuple[Path, dict[str, str]]) -> Counter[str]:
        """Pool worker for the unrouted census."""
        directory, planes = argument
        return SlotTies().lost_of_study(directory, planes)

    @staticmethod
    def _contested_of(argument: tuple[Path, dict[str, str]]) -> dict[str, int]:
        """Pool worker: one study's contested slots.

        A staticmethod rather than a module-level function because the house rule wants behaviour on
        the class that owns it — and a staticmethod still pickles by qualified name, which is what the
        child process needs.
        """
        directory, planes = argument
        return SlotTies().of_study(directory, planes)


@dataclass(frozen=True)
class SlotTiesReport:
    """The two censuses as prose — the rendering half of what `SlotTies` used to be.

    Split for the reason the other reports were: counting walks every study's series on disk, and
    rendering needs only the Counter it produced. Held together the class read as four unconnected
    method clusters, and neither table could be exercised without a corpus to count.
    """

    totals: Counter[str]
    studies: int

    def unrouted(self) -> list[str]:
        """The unrouted census, with the slice count that decides whether it matters."""
        totals, studies = self.totals, self.studies
        series = totals["__series__"]
        lost = sum(count for key, count in totals.items() if not key.startswith("__"))
        share = lost / max(series, 1) * PERCENT
        lines = [f"{studies} studies, {series} series acquired, {lost} unrouted ({share:.1f}%)"]
        for key, count in sorted(totals.items(), key=lambda item: -item[1]):
            if not key.startswith("__"):
                lines.append(f"  {key}: {count}")
        lines.append(f"  slices in unrouted series: {totals['__slices__']}")
        lines.append(
            "  a discarded localiser is worth nothing and a discarded axial T1 may be worth a lot - "
            "the SEQUENCE breakdown is the answer here, not the percentage"
        )
        return lines

    def contested(self) -> list[str]:
        """The counts as lines, with the verdict each one supports stated rather than implied."""
        totals, studies = self.totals, self.studies
        lines = [f"contested slots over {studies} studies:"]
        for slot in SLOTS:
            count = totals[slot.name]
            lines.append(f"  {slot.name}: {count} ({count / max(studies, 1) * PERCENT:.1f}% of studies)")
        affected = totals["studies_with_any"]
        rate = affected / max(studies, 1) * PERCENT
        lines.append(f"  studies with at least one contested slot: {affected} ({rate:.1f}%)")
        lines.append(
            "  a LOW number kills the tie-break hypothesis for the 0.011 leaderboard gap - the two "
            "pipelines would agree almost everywhere, and the residual must come from elsewhere"
        )
        lines.append(
            "  a HIGH number does not prove it either: a tie only matters when the tied series differ "
            "in what they show. That is the next measurement, not this one"
        )
        return lines


def main() -> None:
    """Count contested slots across the training corpus."""
    import argparse  # noqa: PLC0415 - CLI-only, and this module is imported as a library by analyses
    import logging  # noqa: PLC0415 - as above
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415 - as above

    from core.data.corpus.study_cache import StudyCache  # noqa: PLC0415 - as above
    from core.paths import DataRoot  # noqa: PLC0415 - as above

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=str, default="train_series")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--studies", type=int, default=0, help="0 reads the whole corpus")
    parser.add_argument("--lost", action="store_true", help="census the series no slot accepts")
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    raw = DataRoot.from_config(arguments.paths).raw()
    table = raw / arguments.split.replace("_series", "_series.csv")

    # FIRST, because it can make everything below moot in one CSV read: if the two pipelines see the
    # candidates in the same order, no number of contested slots can make them choose differently.
    ties = SlotTies()
    disagreeing, total = ties.disagreeing_order(table)
    logger.info(
        "%d of %d studies order their series differently in the table than on disk (%.1f%%)%s",
        disagreeing,
        total,
        disagreeing / max(total, 1) * PERCENT,
        "" if disagreeing else " — SAME ORDER, so both tie rules pick the same series everywhere",
    )

    planes = StudyCache.planes_of(table)
    directories = sorted(path for path in (raw / arguments.split).iterdir() if path.is_dir())
    if arguments.studies:
        directories = directories[: arguments.studies]

    totals: Counter[str] = Counter()
    worker = SlotTies._lost_of if arguments.lost else SlotTies._contested_of  # noqa: SLF001
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        work = ((directory, planes) for directory in directories)
        for counted in pool.map(worker, work, chunksize=8):
            totals.update({name: count for name, count in counted.items() if count})
            if not arguments.lost:
                totals["studies_with_any"] += int(any(counted.values()))

    report = SlotTiesReport(totals, len(directories))
    lines = report.unrouted() if arguments.lost else report.contested()
    for line in lines:
        logger.info("%s", line)


if __name__ == "__main__":
    main()
