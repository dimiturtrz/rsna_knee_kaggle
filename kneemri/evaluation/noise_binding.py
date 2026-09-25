"""Does a member track the LEXICON or the RADIOLOGIST where the two disagree?

Run as `python -m kneemri.evaluation.noise_binding --checkpoints <--hold-annotations runs>`.

THE QUESTION, AND WHY IT DECIDES A LANE RATHER THAN A KNOB. `core.data.labels.reader_agreement.errors`
measures the label reader's dominant defect: **108 confident false positives over the 58 annotated
studies, all verdict YES, 15.8% of every annotated cell** — against 32 misses in the other direction. That
is a supervision defect four times the size of the silence hole `knee-pju4` was originally written about,
and it is a standing candidate explanation for why every architecture, decode, pooling and regularisation
family in `experiments/README.md` reads flat inside the 0.0062 floor: they are all fitting a target that
says YES on one cell in six where the radiologist says NO.

But a defect in the LABELS is not automatically a ceiling on the MODEL. A member trained on 4406 studies
sees the over-called cells as a minority of noisy targets among many consistent ones, and a fit with
enough regularisation and enough agreeing evidence can average them out — in which case the label lane
buys nothing no matter how large the defect looks in the table. Nothing in this repository has asked which
of those two worlds we are in, and the answer changes what the remaining card time is worth spending on.

THE INSTRUMENT. Take the cells where reader and radiologist agree — positive and negative — as the two
poles, and ask where the DISPUTED cells fall between them:

    tracking = (mean p(over-called) - mean p(agreed negative))
               / (mean p(agreed positive) - mean p(agreed negative))

Unit-free by construction, so it does not care what a member's absolute calibration is:

  * `tracking ~ 1` — the member is as confident on cells the reader INVENTED as on cells that are really
    positive. It learned the lexicon's mistake. The label lane is LIVE and a better reader is a real
    lever.
  * `tracking ~ 0` — the member is as unconvinced on the invented positives as on true negatives. It
    already averaged the noise away from image evidence alone, the 15.8% is a defect the fit does not
    inherit, and the label lane is CLOSED for this defect however large the table makes it look.

THE SAME RATIO IS DEFINED FOR THE MISSES and read the other way round: a `missed` cell is truly positive
and the reader said NO, so a member near 1 there is seeing the finding the reader failed to report.

WHY THIS COSTS NOTHING. It reads checkpoints already on disk through `AnnotatedReadings`, which refuses
any member not trained with `--hold-annotations` — so the annotated cells are held out by construction and
the reading is not the member's own supervision reflected back. Zero submission slots, zero training
hours, and it gates the label work rather than following it.

THE DENOMINATOR, STATED BECAUSE IT IS SMALL AND IT BOUNDS EVERY CLAIM HERE. 58 annotated studies, 1.3% of
the corpus, ~684 cells of which ~108 are over-called and ~32 missed. Cells within a study share a report
and a radiologist and are NOT independent draws, so every interval here is bootstrapped over STUDIES, the
same discipline `reader_agreement` adopts and for the same reason. A ratio built on 108 cells from 58
studies is a DIRECTIONAL reading; it can tell 1 from 0 and it cannot resolve 0.6 from 0.7.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.data.labels.reader_agreement import MIDPOINT, POSITIVE, SILENT, ReaderAgreement
from core.findings import TARGETS

logger = logging.getLogger(__name__)

RESAMPLES = 5000
SEED = 2026
INTERVAL = (2.5, 97.5)

AGREED_POSITIVE = "agreed positive"
OVER_CALLED = "over-called"
AGREED_NEGATIVE = "agreed negative"
MISSED = "missed"
UNREAD = "silent"
"""The five cells a (verdict, truth) pair can fall into. The two AGREED groups are the poles the disputed
groups are measured between; `UNREAD` is carried so the census adds up to the annotated set and is never
scored — an abstention is not a disagreement."""

POLES = (AGREED_NEGATIVE, AGREED_POSITIVE)
DISPUTED = (OVER_CALLED, MISSED)

LEXICON = 1.0
RADIOLOGIST = 0.0
"""What the ratio reads at each of its two interpretable ends, named so the report never prints a bare
number without saying which world it is."""

SEPARATION_FLOOR = 0.05
"""The smallest pole gap the ratio may be computed over. The denominator is a difference of two means; if
a member barely separates cells everyone agrees are positive from cells everyone agrees are negative, the
ratio is dividing by noise and its magnitude means nothing. Refuse rather than report — a member this
weak is not evidence about label noise, it is evidence about the member."""


class UndiscriminatingMemberError(Exception):
    """A member whose agreed-positive and agreed-negative means are too close to divide by."""


@dataclass(frozen=True)
class Binding:
    """Where the disputed cells sit between the two agreed poles, with the census that produced it."""

    ratios: dict[str, float]
    intervals: dict[str, tuple[float, float]]
    means: dict[str, float]
    counts: dict[str, int]

    def verdict(self, group: str) -> str:
        """Which world this group's ratio puts us in, stated rather than left to the reader."""
        ratio = self.ratios[group]
        if ratio >= (LEXICON + RADIOLOGIST) / 2:
            return f"tracks the LEXICON ({ratio:.2f} of the way to a true positive)"
        return f"tracks the RADIOLOGIST ({ratio:.2f} of the way to a true positive)"


class NoiseBinding:
    """Whether the label reader's confident errors reached the model that trained on them."""

    def __init__(self, resamples: int = RESAMPLES, seed: int = SEED) -> None:
        self.resamples = resamples
        self.seed = seed

    def groups(self, cells: pd.DataFrame) -> pd.DataFrame:
        """Label every annotated cell by how the reader and the radiologist related on it.

        The split is on the SAME two predicates `reader_agreement.errors` uses — verdict equal to the
        positive token, truth past the midpoint — so the four counts here reconcile with that module's
        108 and 32 rather than describing a differently-drawn set.
        """
        called = cells["verdict"] == POSITIVE
        positive = cells["truth"] > MIDPOINT
        silent = cells["verdict"] == SILENT
        named = cells.copy()
        named["group"] = np.select(
            [silent, called & positive, called & ~positive, ~called & positive],
            [UNREAD, AGREED_POSITIVE, OVER_CALLED, MISSED],
            default=AGREED_NEGATIVE,
        )
        return named

    def probabilities(
        self,
        cells: pd.DataFrame,
        names: list[str],
        member: Float[NDArray[np.float64], "studies findings"],
    ) -> pd.DataFrame:
        """Attach each cell the member's probability for that (study, finding).

        Joined by NAME on the study and by POSITION in `TARGETS` on the finding, which is the order every
        member's output carries. Cells whose study the member did not read are dropped rather than filled:
        a member is entitled to a corpus that does not contain every annotated study, and inventing a
        probability for one would put a number where there is no reading.
        """
        row = {name: index for index, name in enumerate(names)}
        column = {target: index for index, target in enumerate(TARGETS)}
        present = cells[cells["study"].isin(row)].copy()
        rows = present["study"].map(row).to_numpy(dtype=np.int64)
        columns = present["target"].map(column).to_numpy(dtype=np.int64)
        present["probability"] = member[rows, columns]
        return present

    def tracking(self, scored: pd.DataFrame) -> Binding:
        """The disputed groups' position between the two agreed poles, with study-bootstrapped intervals.

        Refuses a member that does not separate the poles (`SEPARATION_FLOOR`) instead of returning a
        ratio over a denominator indistinguishable from zero.
        """
        means = {str(group): float(frame["probability"].mean()) for group, frame in scored.groupby("group")}
        counts = {str(group): len(frame) for group, frame in scored.groupby("group")}
        missing = [pole for pole in POLES if pole not in means]
        if missing:
            message = f"no cells in {missing} — the poles are what the ratio is measured between"
            raise UndiscriminatingMemberError(message)
        spread = means[AGREED_POSITIVE] - means[AGREED_NEGATIVE]
        if spread < SEPARATION_FLOOR:
            message = (
                f"member separates the agreed poles by only {spread:.4f} (floor {SEPARATION_FLOOR}) — "
                "the ratio would divide by noise, so this member is not evidence about label noise"
            )
            raise UndiscriminatingMemberError(message)
        ratios: dict[str, float] = {}
        intervals: dict[str, tuple[float, float]] = {}
        for group in DISPUTED:
            if group not in means:
                continue
            ratios[group] = (means[group] - means[AGREED_NEGATIVE]) / spread
            intervals[group] = self._resample(scored, group)
        return Binding(ratios=ratios, intervals=intervals, means=means, counts=counts)

    def _resample(self, scored: pd.DataFrame, group: str) -> tuple[float, float]:
        """The ratio's interval, drawing STUDIES with replacement because cells within one are not independent."""
        studies = scored["study"].unique()
        by_study = {study: frame for study, frame in scored.groupby("study")}
        generator = np.random.default_rng(self.seed)
        drawn = []
        for _ in range(self.resamples):
            picked = generator.choice(studies, size=len(studies), replace=True)
            sample = pd.concat([by_study[study] for study in picked])
            pooled = sample.groupby("group")["probability"].mean()
            if not all(pole in pooled for pole in POLES) or group not in pooled:
                continue
            spread = pooled[AGREED_POSITIVE] - pooled[AGREED_NEGATIVE]
            if spread < SEPARATION_FLOOR:
                continue
            drawn.append((pooled[group] - pooled[AGREED_NEGATIVE]) / spread)
        if not drawn:
            return (float("nan"), float("nan"))
        low, high = np.percentile(np.array(drawn), INTERVAL)
        return (float(low), float(high))

    def report(self, binding: Binding) -> list[str]:
        """The census, the two ratios, and the lane each one licenses — never a bare number."""
        lines = ["cells by how reader and radiologist related:"]
        for group, count in sorted(binding.counts.items(), key=lambda item: -item[1]):
            mean = binding.means.get(group, float("nan"))
            lines.append(f"  {group:<18} {count:>4} cells, member mean {mean:.4f}")
        lines.append("")
        low_pole, high_pole = binding.means[AGREED_NEGATIVE], binding.means[AGREED_POSITIVE]
        lines.append(f"poles: agreed negative {low_pole:.4f} -> agreed positive {high_pole:.4f}")
        for group, ratio in binding.ratios.items():
            low, high = binding.intervals[group]
            lines.append(f"  {group}: tracking {ratio:+.3f} [{low:+.3f}, {high:+.3f}] — {binding.verdict(group)}")
        lines.append("")
        lines.append(
            "a tracking near 1 on over-called cells means the member inherited the reader's confident "
            "errors and the label lane is LIVE; near 0 means it averaged them away and the lane is CLOSED "
            "for this defect"
        )
        return lines


def main() -> None:
    """Read the given members over the annotated cells and report whether they inherited the reader's errors."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--checkpoints", type=Path, nargs="+", required=True, help="--hold-annotations runs")
    command.parser.add_argument("--device", default="cpu", choices=("cuda", "cpu"))
    command.parser.add_argument("--labels", type=Path, default=None, help="read table; default report_labels_v2")
    arguments = command.parsed()

    from kneemri.evaluation.annotated_readings import AnnotatedReadings  # noqa: PLC0415 - CLI-only
    from kneemri.evaluation.supervision_gap import SupervisionGap  # noqa: PLC0415 - CLI-only

    root = command.rooted(arguments)
    table = arguments.labels or (root.labels())
    annotated = pd.read_csv(root.train_table())
    read = pd.read_csv(table)
    cells = ReaderAgreement().cells(annotated, read)

    identifiers = SupervisionGap().annotated(root.train_table())[0]
    reader = AnnotatedReadings(root, torch.device(arguments.device))
    names, per_member = reader.readings(arguments.checkpoints, identifiers)

    binding = NoiseBinding()
    grouped = binding.groups(cells)
    for checkpoint, values in per_member:
        scored = binding.probabilities(grouped, names, values)
        command.report([f"=== {checkpoint.name} ===", *binding.report(binding.tracking(scored))])


if __name__ == "__main__":
    main()
