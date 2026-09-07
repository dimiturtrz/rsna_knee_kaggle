"""Per-column areas on the 4406-study corpus surface, and what the 896 frame was reading instead.

    python -m kneemri.evaluation.column_surface --surface teacher_clean_full.npz
    python -m kneemri.evaluation.column_surface --surface teacher_clean_full.npz \
        --against teacher_foreign_corpus.npz --frame <member>.holdout.npz

WHY THE SURFACE IS THE PER-COLUMN RULER FROM NOW ON. Every per-column claim this project has made was
read on the 896-study frame, where Fracture carries 39 positives and a bootstrap half-width of +/-0.067
— a column that cannot resolve anything smaller than a family gap, and a week of column rulings was
aimed with it. The same twelve columns read on the out-of-fold corpus surface carry 192 Fracture
positives and 272 MCL, and the halves shrink 2.6x and 2.1x (researcher D84a, 2026-09-07).

IT IS THE SAME RULER AT LARGER n, NOT A DIFFERENT ONE. Labels rebuilt from `report_labels_v2.csv` and
binarised at the midpoint reproduce the frame's own truth table exactly: agreement 1.0000 over all
10752 cells of the 896 frame, all twelve columns 1.000. That validation is what licenses re-reading an
old frame claim here rather than re-measuring it — and `--frame` prints both readings side by side so a
claim's SHIFT is visible instead of inferred.

WHAT IT CANNOT DO. Only models with five-fold out-of-fold predictions live on this surface — ours and
his, two of them. A member trained on one fold has no honest row here, and putting it on the surface
costs a five-fold training run. So this instrument re-reads the columns of the models that are already
here; it does not price a new member.

THE TWO SURFACES ARE OUT-OF-FOLD UNDER DIFFERENT SPLITS. Each row is honest for its own model, so a
delta between them is a fair model-vs-model comparison per row and NOT a matched-fold one.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.bootstrap import STUDY_RESAMPLES, StudyBootstrap
from core.findings import TARGETS
from kneemri.evaluation.gold_rows import GoldRows
from kneemri.evaluation.scores import LabelScores

logger = logging.getLogger(__name__)

IDENTIFIERS = "identifiers"
SCORES = "scores"
STUDIES = "studies"
"""The key a holdout record spells its identifiers under; a corpus surface spells the same thing
`identifiers`, which is why a frame is read through its own reader and not this module's."""

CORPUS_LABELS = Path("labels") / "report_labels_v2.csv"


@dataclass(frozen=True)
class SurfaceRow:
    """One finding's column on one surface: its area, the positives behind it, and its interval."""

    finding: str
    scored: int
    positives: int
    area: float
    low: float
    high: float

    def half(self) -> float:
        """The half-width the claim is actually resolved to — the number a frame comparison turns on."""
        return (self.high - self.low) / 2.0

    def line(self) -> str:
        """The printed row. The interval is never optional: a bare column area is what the 896 frame
        was quoted as for a week, and the half is the whole reason this surface was reached for."""
        return (
            f"{self.finding:<18} auc {self.area:.4f}  +/-{self.half():.4f}"
            f"  [{self.low:.4f}, {self.high:.4f}]  n={self.scored} pos={self.positives}"
        )


class ColumnSurface:
    """An out-of-fold corpus surface, scored per column against our own labels with an interval."""

    def __init__(self, labels: Path, resamples: int = STUDY_RESAMPLES) -> None:
        self.labels = labels
        self.resamples = resamples
        self.scores = LabelScores()

    @staticmethod
    def read(surface: Path) -> tuple[list[str], Float[NDArray[np.float64], "studies findings"]]:
        """A surface's identifiers and its out-of-fold scores, refusing a file that lacks either.

        Refused rather than defaulted for `ForeignHoldout`'s reason: a table without identifiers can
        only be joined by position, and a positional join of two corpora produces a plausible AUC with
        nothing anywhere to mark it wrong.
        """
        held = np.load(surface, allow_pickle=True)
        missing = [key for key in (IDENTIFIERS, SCORES) if key not in held]
        if missing:
            message = f"{surface} carries no {missing} - it is not a corpus surface"
            raise ValueError(message)
        return [str(identifier) for identifier in held[IDENTIFIERS]], held[SCORES].astype(np.float64)

    @staticmethod
    def onto(
        identifiers: list[str],
        surface: tuple[list[str], Float[NDArray[np.float64], "rows findings"]],
        name: str,
    ) -> Float[NDArray[np.float64], "studies findings"]:
        """That surface's rows in THIS order — the one join in this module, by identifier only.

        A study the surface does not carry is a refusal. Dropping it would silently compare two
        surfaces on two different populations, which is the failure mode a frame comparison exists to
        expose rather than to commit.
        """
        rows, scores = surface
        index = {identifier: position for position, identifier in enumerate(rows)}
        missing = [identifier for identifier in identifiers if identifier not in index]
        if missing:
            message = f"{name} has no row for {len(missing)} of {len(identifiers)} studies, first {missing[0]}"
            raise ValueError(message)
        return scores[[index[identifier] for identifier in identifiers]]

    def truth(self, identifiers: list[str]) -> Float[NDArray[np.float32], "studies findings"]:
        """Our report labels for exactly these studies, read raw — the frame's own truth table at
        larger n, validated cell for cell against it (1.0000 over 10752 cells)."""
        return GoldRows().reported(self.labels, identifiers)

    def rows(
        self,
        scores: Float[NDArray[np.float64], "studies findings"],
        truth: Float[NDArray[np.float32], "studies findings"],
    ) -> list[SurfaceRow]:
        """One row per finding, in the metric's own column order, each with its bootstrap interval.

        Column order rather than sorted by area: this report is read against claims already written
        down per finding, and a re-sort between two frames would hide the shift it exists to show.
        """
        areas = self.scores.per_label(scores, truth)
        intervals = StudyBootstrap(self.resamples).per_label(
            len(scores), lambda drawn: self.scores.per_label(scores[drawn], truth[drawn])
        )
        positives = (truth > LabelScores().cut).sum(axis=0)
        return [
            SurfaceRow(
                finding=finding,
                scored=len(scores),
                positives=int(positives[column]),
                area=areas[finding],
                low=intervals[finding][0],
                high=intervals[finding][1],
            )
            for column, finding in enumerate(TARGETS)
        ]

    def delta(
        self,
        first: Float[NDArray[np.float64], "studies findings"],
        second: Float[NDArray[np.float64], "studies findings"],
        truth: Float[NDArray[np.float32], "studies findings"],
    ) -> dict[str, tuple[float, float, float]]:
        """Per column, `second - first` with its PAIRED interval — both surfaces on the same rows.

        Paired because the two share every study's difficulty and the difference does not care about
        it; unpaired halves added in quadrature would bound a column at twice what it is resolved to.
        """
        points = self.scores.per_label(second, truth)
        firsts = self.scores.per_label(first, truth)
        bounds = self.scores.paired_interval(first, second, truth, self.resamples)
        return {
            finding: (points[finding] - firsts[finding], bounds[finding][0], bounds[finding][1]) for finding in TARGETS
        }

    @staticmethod
    def frame_studies(record: Path) -> list[str]:
        """The studies a holdout record scored — the 896-frame population, read off the frame itself.

        Off the record and never re-derived: a frame reconstructed from a fold file is a frame before
        alignment dropped its unshared rows, and comparing 909 studies against a claim made on 896 is
        the same population error this module refuses everywhere else.
        """
        held = np.load(record, allow_pickle=True)
        if STUDIES not in held:
            message = f"{record} carries no {STUDIES!r} - it is not a holdout record"
            raise ValueError(message)
        return [str(identifier) for identifier in held[STUDIES]]

    @staticmethod
    def shift(corpus: list[SurfaceRow], frame: list[SurfaceRow]) -> list[str]:
        """Both readings of each column, side by side — the re-read this instrument exists for.

        The last field is what the frame's own half-width was, so a claim that moved by less than the
        frame could resolve is visible as a claim the frame never made rather than as a correction.
        """
        lines = [f"{'finding':<18} {'corpus':>8} {'+/-':>7}   {'frame':>8} {'+/-':>7}   {'shift':>8}  frame pos"]
        for wide, narrow in zip(corpus, frame, strict=True):
            lines.append(
                f"{wide.finding:<18} {wide.area:>8.4f} {wide.half():>7.4f}   "
                f"{narrow.area:>8.4f} {narrow.half():>7.4f}   {wide.area - narrow.area:>+8.4f}"
                f"  pos={narrow.positives}"
            )
        return lines

    def report(  # devtools-ignore: test-mirror
        self, surface: Path, against: Path | None, frame: Path | None
    ) -> list[str]:
        """The surface's columns, optionally beside a second surface and beside the 896 frame."""
        rows, scores = self.read(surface)
        truth = self.truth(rows)
        lines = [f"{surface.name}: {len(rows)} studies", *[row.line() for row in self.rows(scores, truth)]]
        if against is not None:
            other = self.onto(rows, self.read(against), against.name)
            lines.append(f"paired delta, {against.name} minus {surface.name}, on the same {len(rows)} rows")
            lines += [
                f"{finding:<18} {point:+.4f}  [{low:+.4f}, {high:+.4f}]"
                for finding, (point, low, high) in self.delta(scores, other, truth).items()
            ]
        if frame is not None:
            held = self.frame_studies(frame)
            narrow = self.onto(held, (rows, scores), surface.name)
            lines.append(f"the {len(held)}-study frame of {frame.name}, same surface, same truth")
            lines += self.shift(self.rows(scores, truth), self.rows(narrow, self.truth(held)))
        return lines


def main() -> None:
    """Read a corpus surface's columns, and re-read a frame claim against them."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--surface", type=Path, required=True, help="an out-of-fold corpus npz")
    command.parser.add_argument("--against", type=Path, default=None, help="a second surface, for a paired delta")
    command.parser.add_argument("--frame", type=Path, default=None, help="a holdout record whose studies to re-read")
    command.parser.add_argument("--labels", type=Path, default=None, help="truth table; report_labels_v2 by default")
    arguments = command.parsed()

    root = command.rooted(arguments)
    labels = arguments.labels or root.processed() / CORPUS_LABELS
    surface = ColumnSurface(labels)
    command.report(surface.report(arguments.surface, arguments.against, arguments.frame))


if __name__ == "__main__":
    main()
