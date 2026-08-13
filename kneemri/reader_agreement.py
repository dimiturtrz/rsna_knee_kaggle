"""Does the label reader agree with the radiologist equally in every language? `python -m kneemri.reader_agreement`.

Every score this project reports is agreement with LLM-read report labels, and a third of the cells the
reader calls positive are not positive to the radiologist. The obvious suspect is language: 59.9% of the
reports are not English (`report_language`), and "the rest needs a multilingual encoder" is the premise
branch (a) of `knee-mxj` rests on. It had never been measured.

IT IS ALSO THE ACCEPTANCE TEST FOR ANY FUTURE READER. A replacement reader has to agree with the
radiologist more than this one does, and a reader built to fix a language has to close a gap in that
language. Both questions are this module, run before and after.

THE DENOMINATOR IS 58 STUDIES — 1.32% of the corpus, all-or-nothing annotated, and the only ground truth
that is not itself the reader's output. Cells within one study share a report and a radiologist, so they
are NOT independent draws: the interval here is bootstrapped over STUDIES, not cells, and it is about
twice the width a cell-wise interval would report.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from kneemri.model import TARGETS
from kneemri.report_language import ReportLanguages

logger = logging.getLogger(__name__)

RESAMPLES = 5000
SEED = 2026
INTERVAL = (2.5, 97.5)
ENGLISH = "en"
POSITIVE = "YES"


class ReaderAgreement:
    """How often the reader's verdict matches the radiologist, split by the report's language."""

    def __init__(self, languages: ReportLanguages | None = None) -> None:
        self.languages = languages or ReportLanguages()

    def cells(self, annotated: pd.DataFrame, read: pd.DataFrame) -> pd.DataFrame:
        """One row per (study, finding) the radiologist annotated AND the reader gave a verdict on.

        Both sides are required. A cell the reader could not read is not a cell it got wrong, and
        counting it as one would measure coverage while claiming to measure accuracy.
        """
        verdicts = [column for column in read.columns if column.endswith("__verdict")]
        merged = annotated.merge(read[["StudyInstanceUID", *verdicts]], on="StudyInstanceUID")
        rows = []
        for _, study in merged.iterrows():
            language = self.languages.language(str(study["Report"]))
            for target in TARGETS:
                verdict, truth = study.get(f"{target}__verdict"), study.get(target)
                if pd.isna(verdict) or pd.isna(truth):
                    continue
                rows.append((study["StudyInstanceUID"], language, str(verdict).upper(), float(truth)))
        return pd.DataFrame(rows, columns=["study", "language", "verdict", "truth"])

    def gap(self, cells: pd.DataFrame) -> dict[str, float]:
        """English minus non-English agreement on the reader's POSITIVE calls, with a study-clustered interval.

        Reported as an interval rather than a verdict because with 58 annotated studies the honest answer
        is a bound, not a number: what this can establish is the size of gap it would have SEEN.
        """
        positives = cells[cells["verdict"] == POSITIVE]
        english = positives[positives["language"] == ENGLISH]
        other = positives[positives["language"] != ENGLISH]
        deltas = self._resample(positives, english["study"].unique(), other["study"].unique())
        low, high = np.percentile(deltas, INTERVAL)
        return {
            "english": float(english["truth"].mean()),
            "other": float(other["truth"].mean()),
            "delta": float(english["truth"].mean() - other["truth"].mean()),
            "low": float(low),
            "high": float(high),
        }

    def _resample(
        self, positives: pd.DataFrame, english: NDArray[np.object_], other: NDArray[np.object_]
    ) -> NDArray[np.float64]:
        by_study = {study: group["truth"].to_numpy() for study, group in positives.groupby("study")}
        generator = np.random.default_rng(SEED)

        def drawn(studies: NDArray[np.object_]) -> float:
            picked = generator.choice(studies, len(studies))
            return float(np.concatenate([by_study[study] for study in picked]).mean())

        return np.array([drawn(english) - drawn(other) for _ in range(RESAMPLES)])

    def report(self, cells: pd.DataFrame) -> list[str]:
        """Per-language agreement on the reader's positive calls, then the English/non-English gap."""
        positives = cells[cells["verdict"] == POSITIVE]
        lines = [f"{len(cells)} annotated cells, {len(positives)} of them called positive by the reader"]
        for language, group in positives.groupby("language"):
            lines.append(f"  {language:>8} {len(group):>4} cells   radiologist agrees {group['truth'].mean():>6.1%}")
        gap = self.gap(cells)
        lines.append(
            f"English {gap['english']:.1%} against non-English {gap['other']:.1%}: {gap['delta']:+.3f} "
            f"[{gap['low']:+.3f}, {gap['high']:+.3f}] over studies"
        )
        return lines


def main() -> None:
    """Report reader-radiologist agreement per language on the annotated studies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, help="the reader's table; defaults to report_labels_v2.csv")
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from core.paths import DataRoot  # noqa: PLC0415 - CLI-only, and this module is imported as a library

    root = DataRoot.from_config(arguments.paths)
    truth = pd.read_csv(root.raw() / "train.csv")
    read = pd.read_csv(arguments.labels or root.processed() / "labels" / "report_labels_v2.csv")
    annotated = truth.dropna(subset=TARGETS, how="all")[["StudyInstanceUID", "Report", *TARGETS]]

    agreement = ReaderAgreement()
    for line in agreement.report(agreement.cells(annotated, read)):
        logger.info(line)


if __name__ == "__main__":
    main()
