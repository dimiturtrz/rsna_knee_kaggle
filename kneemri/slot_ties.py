"""How often two series compete for one slot — the ambiguity our routing resolves by list order.

`SlotScheme.assign` picks the densest stack for a slot and breaks a tie with `max`, which keeps the
FIRST maximal element in list order. The published members we grafted were routed by a pandas
`sort_values` that keeps the first in FRAME order. Where a slot is contested the two rules can pick
different series, and a different series in a slot is a different image under the head — nothing
downstream recovers it.

Two independent measurements say that costs us something: our pipeline reproduces those members'
out-of-fold predictions only to max|diff| 0.025, and the graft scored 0.888 against the public forks'
~0.899 on identical weights. This module measures how often the ambiguity ARISES at all. Rare ties
with the residual intact would kill the hypothesis, which is most of what it is worth measuring.

It reads one DICOM header per series and no pixels, so it is cheap and runs happily beside training.
"""

from collections import Counter
from pathlib import Path

from kneemri.slots import SLOTS, SlotScheme
from kneemri.study import Study

PERCENT = 100.0


class SlotTies:
    """Counts contested slots across a corpus of studies."""

    def __init__(self, scheme: SlotScheme | None = None) -> None:
        self.scheme = scheme or SlotScheme()

    def of_study(self, directory: Path, planes: dict[str, str]) -> dict[str, int]:
        """Which of this study's slots were won by a tie on slice count."""
        study = Study.from_directory(directory, planes)
        return self.scheme.contested(study.acquisitions, planes)

    def report(self, totals: Counter[str], studies: int) -> list[str]:
        """The counts as lines, with the verdict each one supports stated rather than implied."""
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

    @staticmethod
    def _contested_of(argument: tuple[Path, dict[str, str]]) -> dict[str, int]:
        """Pool worker: one study's contested slots.

        A staticmethod rather than a module-level function because the house rule wants behaviour on
        the class that owns it — and a staticmethod still pickles by qualified name, which is what the
        child process needs.
        """
        directory, planes = argument
        return SlotTies().of_study(directory, planes)


def main() -> None:
    """Count contested slots across the training corpus."""
    import argparse  # noqa: PLC0415 - CLI-only, and this module is imported as a library by analyses
    import logging  # noqa: PLC0415 - as above
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415 - as above

    from core.paths import DataRoot  # noqa: PLC0415 - as above
    from kneemri.study_cache import StudyCache  # noqa: PLC0415 - as above

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=str, default="train_series")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--studies", type=int, default=0, help="0 reads the whole corpus")
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    raw = DataRoot.from_config(arguments.paths).raw()
    planes = StudyCache.planes_of(raw / arguments.split.replace("_series", "_series.csv"))
    directories = sorted(path for path in (raw / arguments.split).iterdir() if path.is_dir())
    if arguments.studies:
        directories = directories[: arguments.studies]

    ties = SlotTies()
    totals: Counter[str] = Counter()
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        work = ((directory, planes) for directory in directories)
        for contested in pool.map(SlotTies._contested_of, work, chunksize=8):  # noqa: SLF001
            totals.update({name: count for name, count in contested.items() if count})
            totals["studies_with_any"] += int(any(contested.values()))

    for line in ties.report(totals, len(directories)):
        logger.info("%s", line)


if __name__ == "__main__":
    main()
