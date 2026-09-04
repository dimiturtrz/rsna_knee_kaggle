"""WHERE in the holdout the board's ordering lives: `python -m kneemri.evaluation.transfer_strata`.

`board_transfer` asks whether our holdout orders entries the way the leaderboard does, over the whole
909 studies at once, and answers with one median ratio. That number is now doing work it cannot carry.
Seven entries on this board are fold-0-honest end to end, and read as a ladder they say two things:

    spansolo_v1   0.8545 -> 0.921        peak8         0.8704 -> 0.932
    holdsolo_v1   0.8634 -> 0.928        foldfour_v1   0.8711 -> 0.936
    holdfive_v1   0.8729 -> 0.936        greedyseat_v1 0.8740 -> 0.936
    honest12      0.8751 -> 0.937

End to end that is 0.015 of board per 0.0184 of holdout, a slope of 0.82. And the last five rows span
+0.0047 of holdout and land inside a single board tick. So the ladder does not compress uniformly: it
transfers steeply at the bottom and flattens at the top, and one median over all pairs reports the
average of two regimes as if it were a constant.

WHY THAT IS THE WRONG SHAPE TO PLAN WITH. Two readings fit those seven rows equally well. Either
transfer is uniformly ~0.8, in which case another +0.005 of holdout is worth +0.004 of board and the
next arm is worth running; or transfer is near 1 low down and ~0 in a band our last five entries all
sit inside, in which case the holdout is BLIND above 0.870 and every plan that ends in "improve the
offline number" is unfunded. Those predict opposite things about the same next experiment.

WHAT THE FIELD CALLS THIS. Recht et al. (arXiv:1806.00451, ICML 2019) measure the same quantity on
CIFAR-10.1 and ImageNetV2 and report slopes ABOVE one — 1.62 and 1.11 — which they read as evidence
against test-set adaptivity. Ours runs the other way, so adaptive overfitting is not the story here.
Roelofs et al. (NeurIPS 2019, a meta-analysis over 120 Kaggle competitions) describe the plateau shape
directly: later submissions move on one axis while stagnating on the other. Akhtar et al. (ICML 2026,
arXiv:2602.16763) name the end state BENCHMARK SATURATION and separate it from stagnation — models
becoming indistinguishable, near a ceiling, which is what +0.0047 of holdout buying +0.0005 of board
looks like.

WHAT THIS MODULE DOES, AND WHY IT COSTS NO SLOT. It cuts the holdout into difficulty strata and asks
the transfer question inside each one. The three candidate causes of a sub-unity slope make different
predictions about that cut:

  * DISTRIBUTION GAP — the hidden test is a harder mix than our holdout. Then the board's ordering is
    recovered in the HARD stratum and the compression is an artefact of scoring on studies the board is
    not weighting the way we are.
  * SATURATION — the metric has no headroom left. Then the compression is worst in the EASY stratum,
    where every entry is already at ceiling and nothing can separate them.
  * NEITHER — every stratum reads a slope near one and the effect is about case weighting or label
    structure rather than about difficulty at all.

DIFFICULTY IS MEASURED WITHOUT THE LABELS, on purpose. The stratum ruler is member DISAGREEMENT: per
study, the spread across a reference pool of what the members believe, meaned over findings. A
truth-aware ruler (per-study residual) would cut the holdout by who got it wrong, and the hardest
stratum would then be mostly errors by construction — an AUC near chance in every entry, which
separates nothing and would read as saturation whatever was true. Disagreement is the epistemic term
`uncertainty_split` already reports, and it is a property of the pool rather than of the answer.

THE REFERENCE POOL IS FIXED ACROSS ENTRIES and is the union of every member any honest entry seats.
Cutting each entry on its OWN disagreement would give every entry a different Q5, and a slope across
entries scored on different studies is not a slope.

WHAT IT CANNOT SAY. The strata are ~180 studies each, so a per-label AUC inside one is wide and a rare
finding may have no positives at all — those labels come back NaN and are dropped from that stratum's
mean, which is why the report prints how many of the twelve were scoreable. The entries also span days
and a kit that changed underneath them, exactly as `board_transfer` states; stratifying does not repair
that, it only asks the confounded question in a place where the two hypotheses differ.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from numpy.typing import NDArray

from core.findings import TARGETS
from core.paths import DataRoot
from kneemri.evaluation.board_transfer import DELTA_FLOOR, ENSEMBLE, RECORD, RESOLVED, BoardEntry, BoardPair
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.record_mix import RecordMix
from kneemri.evaluation.scores import LabelScores
from kneemri.inference.vote import LogitAverage

logger = logging.getLogger(__name__)

STRATA = 5
"""How many difficulty bands the holdout is cut into. Five over 909 studies is ~180 a band, which is the
smallest stratum in which a twelve-label mean is worth printing at all."""

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "paths.yaml"


@dataclass(frozen=True)
class Stratum:
    """One difficulty band of the holdout: which rows it holds, and how hard they are."""

    band: int
    rows: Int[NDArray[np.int64], " studies"]
    difficulty: float
    """The band's mean member disagreement — the quantity it was cut on, printed so the bands can be
    read as a scale rather than as five anonymous buckets."""

    def named(self) -> str:
        """`Q1` is the easiest band. Numbered from one because the report is read by people."""
        return f"Q{self.band + 1}"


@dataclass(frozen=True)
class StratumReading:
    """What one entry scored inside one band, beside the board number it actually got."""

    entry: str
    band: int
    offline: float
    board: float
    scoreable: int
    """How many of the twelve findings had both classes present in this band. A band that loses a rare
    finding is scoring a different metric from the whole-holdout number, and the reader must see it."""


class TransferStrata:
    """The offline-to-board slope, asked separately inside each difficulty band of the holdout."""

    def __init__(self, models: Path, record: Path = RECORD, scope: str = ENSEMBLE, bands: int = STRATA) -> None:
        self.models = models
        self.record = record
        self.scope = scope
        self.bands = bands
        self.scores = LabelScores()

    def rows(self) -> list[dict[str, object]]:
        """Every recorded line, exactly as the board file holds it."""
        return [json.loads(line) for line in self.record.read_text(encoding="utf-8").splitlines() if line]

    def honest(self) -> list[dict[str, object]]:
        """The entries this reading can rebuild: both numbers, this scope, and every member on disk.

        A row naming a member with no holdout record is DROPPED rather than scored on what it does
        have. Every `--fold -1` member is exactly that case — it held nothing out, so it left no record
        — which means the disk check enforces the honesty rule without needing to know the naming
        convention that spells it.
        """
        return [row for row in self.rows() if self._rebuildable(row)]

    def _rebuildable(self, row: dict[str, object]) -> bool:
        """Whether this row carries both numbers on this scope and every member it names is on disk."""
        named = row.get("members_named")
        if row.get("scope") != self.scope or row.get("offline") is None or row.get("board") is None:
            return False
        if not isinstance(named, list) or not named:
            return False
        return all(path.exists() for path in self._checkpoints([str(name) for name in named]))

    def _checkpoints(self, names: list[str]) -> list[Path]:
        """Bare member names as the checkpoint paths their holdout records sit beside."""
        return [self.models / f"{name}.pt" for name in names]

    def _pool(self, names: list[str]) -> Pool:
        """These members' aligned predictions, combined the way a submission combines them.

        `LogitAverage` rather than the ledger's default rank-average, because that is what every entry
        in this ladder actually served — the ruler and the entry disagreeing on the vote rule is the
        harness gap `Pool.vote` exists to close.
        """
        return RecordMix(self._checkpoints(names)).per_member(LogitAverage())

    def members(self) -> list[str]:
        """Every member any rebuildable entry seats, once — the reference pool the cut is made on."""
        seen: list[str] = []
        for row in self.honest():
            named = row.get("members_named")
            for name in named if isinstance(named, list) else []:
                if str(name) not in seen:
                    seen.append(str(name))
        return seen

    def disagreement(self) -> Float[NDArray[np.float64], " studies"]:
        """Per study, how much the reference pool disagrees about it, meaned over the twelve findings.

        The spread across members of what they believe — no labels are read, so the cut is a property
        of the pool rather than of the answer. See the module docstring for why a truth-aware ruler
        would build a hardest stratum that separates nothing.
        """
        pool = self._pool(self.members())
        stacked = np.stack(pool.predictions)
        return np.asarray(stacked.std(axis=0).mean(axis=1), dtype=np.float64)

    def strata(self) -> list[Stratum]:
        """The holdout cut into equal-count difficulty bands, easiest first."""
        spread = self.disagreement()
        order = np.argsort(spread, kind="mergesort")
        return [
            Stratum(band=band, rows=rows, difficulty=float(spread[rows].mean()))
            for band, rows in enumerate(np.array_split(order, self.bands))
        ]

    def readings(self) -> list[StratumReading]:
        """Every rebuildable entry scored inside every band, beside the board number it was given."""
        bands = self.strata()
        found: list[StratumReading] = []
        for row in self.honest():
            named = row.get("members_named")
            names = [str(name) for name in named] if isinstance(named, list) else []
            pool = self._pool(names)
            whole = list(range(len(pool)))
            for stratum in bands:
                band = pool.rows(stratum.rows)
                per_label = self.scores.per_label(band.combined(whole), band.truth)
                values = [value for value in per_label.values() if not np.isnan(value)]
                found.append(
                    StratumReading(
                        entry=str(row["id"]),
                        band=stratum.band,
                        offline=float(np.mean(values)) if values else float("nan"),
                        board=float(str(row["board"])),
                        scoreable=len(values),
                    )
                )
        return found

    def compression(self, readings: list[StratumReading]) -> float:
        """The median board-per-holdout ratio over the pairs this band resolves, or NaN when it has none.

        The median over pairs, and the pairs are `board_transfer`'s: the entries are neither independent
        nor matched, so a fitted line would report the two extreme entries with a precision the data
        does not have. `RESOLVED` is that module's floor for "the holdout separated these two at all".
        """
        entries = [BoardEntry(id=r.entry, offline=r.offline, board=r.board, members=0) for r in readings]
        ordered = sorted(entries, key=lambda entry: entry.offline)
        ratios = [
            BoardPair(better=better, worse=worse).ratio()
            for index, worse in enumerate(ordered)
            for better in ordered[index + 1 :]
            if better.offline - worse.offline >= max(DELTA_FLOOR, RESOLVED)
        ]
        return float(np.median(ratios)) if ratios else float("nan")

    def lines(self) -> list[str]:
        """The report: one block per band, and the whole-holdout row it has to be read against."""
        readings = self.readings()
        entries = sorted({reading.entry for reading in readings})
        out = [
            f"{len(entries)} rebuildable {self.scope} entries over {self.bands} difficulty bands, "
            f"cut on member disagreement across {len(self.members())} reference members",
            f"entries: {', '.join(entries)}",
            "",
        ]
        for stratum in self.strata():
            band = [reading for reading in readings if reading.band == stratum.band]
            ratio = self.compression(band)
            scoreable = min(reading.scoreable for reading in band) if band else 0
            out.append(
                f"{stratum.named()}  {len(stratum.rows):>4} studies  disagreement {stratum.difficulty:.4f}  "
                f"slope {ratio:+.2f}  ({scoreable} of {len(TARGETS)} findings scoreable)"
            )
            out.extend(
                f"    {reading.entry:<20} offline {reading.offline:.4f}  board {reading.board:.3f}"
                for reading in sorted(band, key=lambda reading: reading.offline)
            )
        return out


def main() -> None:
    """Print the per-band transfer report."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=RECORD, help="the board file to read entries from")
    parser.add_argument("--scope", default=ENSEMBLE, help="which board scope to read")
    parser.add_argument("--bands", type=int, default=STRATA, help="how many equal-count difficulty bands")
    parser.add_argument("--models", type=Path, default=None, help="where the checkpoints and their records live")
    arguments = parser.parse_args()
    models = arguments.models or DataRoot.from_config(CONFIG).models()
    for line in TransferStrata(models, arguments.record, arguments.scope, arguments.bands).lines():
        logger.info(line)


if __name__ == "__main__":
    main()
