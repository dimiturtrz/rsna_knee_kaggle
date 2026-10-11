"""The ledger's second ruler: `python -m kneemri.evaluation.ledger75 --holdouts <dir>`.

WHY A SECOND NUMBER AND NOT A NEW DEFAULT. Every number in the ledger was read with graded truth cut at
the midpoint, so a cut that moved globally would silently reprice the whole record. This prints the same
members under both cuts, side by side, and changes no existing semantics: `ledger_check`,
`convergence_check` and `roster_check` still read the midpoint macro they always read.

THE ARGUMENT FOR THE 0.75 CUT (researcher D25, 30 members, 896 rows against the gold-58). The label
reader's 0.68 rung means "possible", and the midpoint counts every one of those rows as a POSITIVE. On
Effusion that rung is 48% gold-positive over 27 rows, so the midpoint ruler was ranking members by how
well they found SMALL FLUID that the radiologist does not call effusion. Recut at 0.75 the LEDGER~gold
Spearman on Effusion goes -.17 -> +.74 (hold -.16 -> +.77), and the macro ordering +.59 -> +.80 (hold
+.25 -> +.71). The eight columns that already had a ruler are the control: mean |move| .18 against .47
on the four that did not.

TWO COLUMNS KEEP THE MIDPOINT, and they are not an exception bolted on to make a number look better.
Medial Meniscus (+.46 -> +.12) and Baker's (+.47 -> +.04) LOSE their ruler at 0.75, because their 0.68
rung carries real positives (n 5 and 6 of the gold set). A column's cut belongs where that column's
coin-flip rung sits, and for those two it sits below 0.75.

WHAT THIS DOES NOT FIX, so nobody reads the macro as a repair of the whole table: Synovitis keeps its
mass at the UNKNOWN 0.28 rung (41 of 57 rows, below every cut), Contusion's coin-flip rung is 0.82 which
is ABOVE every cut so a recut only makes it worse, and Fracture has no coin-flip rung at all. Those three
are a different mechanism and this ruler is silent about them.

Board evidence (researcher D27): across ~530 leaderboard pairs LEDGER75 leaves 5 inversions where the
midpoint leaves 21, and 0 of 326 at the >=.0053 floor; R^2 .80 -> .96 with one slope 1.16-1.22 where the
midpoint's wanders 1.05-1.41.
"""

import hashlib
import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import numpy as np
import torch

from core import cli
from core.findings import TARGETS
from kneemri.evaluation.member_pool import DECLINES, MemberDeclines, MemberPool
from kneemri.evaluation.scores import MIDPOINT, LabelScores
from kneemri.runs.artefact import Artefact
from kneemri.runs.holdout_record import SUFFIX, HoldoutReading, HoldoutRecord

logger = logging.getLogger(__name__)

RECUT = 0.75
"""Where the reader's "possible" rung (0.68) stops counting as a positive. D25's measured value, not a
tuned one: it is the only cut that separates 0.68 from 0.82 given the five levels v2 actually takes."""

MIDPOINT_COLUMNS: tuple[str, ...] = ("Medial Meniscus", "Baker's")
"""The two findings whose 0.68 rung carries real positives, so the recut costs them their ruler."""


class Ruler(StrEnum):
    """The two cuts an offline number is read under, by the name every CLI and record field uses."""

    MIDPOINT = "midpoint"
    LEDGER75 = "ledger75"

    @property
    def column(self) -> str:
        """The `comp/board.jsonl` field a number read under this ruler is recorded in."""
        return RULINGS[self].column

    def scorer(self) -> LabelScores:
        """A fresh scorer applying this ruler's cut, for a caller holding arrays rather than a record."""
        return RULINGS[self].scorer()


@dataclass(frozen=True)
class Ruling:
    """What one ruler IS: the field its numbers are recorded in and the scorer that produces them.

    One row per ruler rather than one table per attribute, so a ruler can never gain a column without
    a cut or a cut without a column - the two parallel maps this replaced could drift apart silently.
    """

    column: str
    scorer: Callable[[], LabelScores]


@dataclass(frozen=True)
class Ledger75Row:
    """One member under both rulers — the pair, never either number alone."""

    name: str
    ledger: float
    ledger75: float
    epoch: int
    studies: int
    between_cut: int = -1
    population: str = ""
    """Digest of the study ids this row was read on — the only thing that licenses comparing two rows.

    An AUC is a statistic OF a study set, so two rows read on different sets are two different questions
    and their difference is not a delta. 426 records on disk carry twelve distinct populations: 375 read
    the 909 set the project ships against, and the rest are fold members that each held out a different
    quarter, near-909 foreign refits, and 60-study smokes. Keyed on the id digest and not on the row
    COUNT, which happens to separate all twelve here and would stop doing so the moment a recut produced
    a second set of the same size.
    """

    void: bool = False
    """Does `comp/members.declines` say this member's number is not a reading of held-out studies?

    This instrument prices whatever holdout file it is handed and cannot see who trained on those
    studies, so a member fitted on its own holdout reads HIGH and sorts to the top. The four highest
    rows of a solo ranking taken on 2026-09-17 were all of that kind, already ruled 98.6% in-sample at
    `comp/members.declines:815-821`, and read as printed the table put a foreign member 0.037 above the
    project's best solo where the board puts the same member 0.021 below it.
    """

    def moved(self) -> float:
        """LEDGER75 minus the midpoint macro. A member's own re-pricing, not a comparison to anything."""
        return self.ledger75 - self.ledger

    def price(self, checkpoint: Path) -> bool:
        """Write this row's midpoint macro into a checkpoint whose own `holdout` is missing or NaN.

        A WEIGHT SOUP IS SCOREABLE BUT UNSHIPPABLE WITHOUT IT. `weight_soup` sets `holdout = nan`, and
        `package_members` reads that field into the manifest row seat ranking is done on (knee-emyky).
        The number written is `ledger`, this row's own — never a second arithmetic for the same field.

        Here and not in `holdout_backfill`, which is the pass that stamps the soup's other missing field:
        pricing a record is this package's arithmetic, and inference may not import the layer that
        scores it. A checkpoint already carrying a number is left alone, because that number is what its
        own training run measured and this ruler must not silently replace a member's recorded price.
        """
        # A record whose checkpoint is gone is priced and unshippable both, and 83 of the members on
        # disk are exactly that (`roster_check`); a sweep must not die on the first one.
        if not checkpoint.exists():
            return False
        saved = dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        if not np.isnan(float(cast(float, saved.get("holdout", np.nan)))):
            return False
        saved["holdout"] = self.ledger
        staged = checkpoint.with_suffix(".pricing")
        torch.save(saved, staged)
        staged.replace(checkpoint)
        return True

    def recut_blind(self) -> bool:
        """Is this row's PAIR of numbers empty of information — no truth mass between the two cuts?

        A record whose truth never lands in `[MIDPOINT, RECUT)` is scored identically by both rulers, so
        `moved()` is 0.0000 BY CONSTRUCTION rather than as a measurement, and reading it as "the recut does
        not touch this member" is the `knee-f9aao` error one level down: a number whose population cannot
        express the question. Researcher D245 is the source, and its criterion is deliberately the truth
        mass and NOT a level count — the twenty-four five-level `*_smoke` records are fully recut-sensitive,
        so counting levels would have refused exactly the wrong set.

        The refusal lives HERE, on the row, and not on `macro()` or `scorer()`: a pool's combined score is
        a legitimate number even when one member in it is blind, and a refusal at the scorer layer would
        void the whole pool for one probe.
        """
        return self.between_cut == 0

    def line(self) -> str:
        """The row as the table prints it, with the VOID and BLIND markers that stop it being quoted bare."""
        markers = (
            (self.void, "  VOID - not a reading of held-out studies"),
            (self.recut_blind(), "  BLIND - no truth between the cuts, move is 0 by construction"),
        )
        return (
            f"{self.name[:40]:40s} {self.ledger:8.4f} {self.ledger75:9.4f} {self.moved():+8.4f} {self.epoch:6d} "
            f"{self.studies:6d}" + "".join(text for flagged, text in markers if flagged)
        )


class LedgerSeventyFive:
    """The mixed-cut ruler: ten findings at 0.75, two at the midpoint, scored off the kept arrays."""

    def __init__(
        self,
        recut: float = RECUT,
        midpoint_columns: tuple[str, ...] = MIDPOINT_COLUMNS,
        declines: MemberDeclines | None = None,
    ) -> None:
        self.recut = LabelScores(recut)
        self.midpoint = LabelScores(MIDPOINT)
        self.midpoint_columns = midpoint_columns
        # Empty rather than reading the file, so the scorer stays pure and the POLICY lives at the
        # entrypoint that already resolves `--declines`. A default that read disk would put a repo file
        # behind every construction, the caller's cwd included.
        self.void = declines.void() if declines else set[str]()

    def scorer(self) -> LabelScores:
        """This ruler as a plain scorer, for a caller holding ARRAYS rather than a member's record.

        `scores.py` records the gap this closes: the mixed cut lived only inside this class, which takes
        a `HoldoutReading`, so no scorer-taking caller — `seat_value`, `member_curve`, `member_distance`,
        `seat_add` — could ask for LEDGER75 at all and every one of them silently read the midpoint. That
        is the mechanism behind the whole cut seam (knee-f9aao): not that 0.5 was chosen, but that it was
        the only thing reachable. The cut map is derived from this instrument's own two fields, so a
        caller can never hold a LEDGER75 that has drifted from the one `macro` reads.
        """
        return LabelScores(self.recut.cut, dict.fromkeys(self.midpoint_columns, MIDPOINT))

    def reading(self, path: Path) -> HoldoutReading:
        """One member's kept predictions and the truth they were scored against."""
        return HoldoutRecord(path).required()

    def per_label(self, reading: HoldoutReading) -> dict[str, float]:
        """The twelve AUCs under the mixed cut, each column read where its own coin-flip rung sits."""
        recut = self.recut.per_label(reading.predictions, reading.truth)
        midpoint = self.midpoint.per_label(reading.predictions, reading.truth)
        return {target: midpoint[target] if target in self.midpoint_columns else recut[target] for target in TARGETS}

    def macro(self, reading: HoldoutReading) -> float:
        """The mean of the twelve, NaN-skipping exactly as the ledger's own macro does."""
        return float(np.nanmean(list(self.per_label(reading).values())))

    def row(self, path: Path) -> Ledger75Row:
        """One member under both rulers, with the epoch and the row count that priced it."""
        reading = self.reading(path)
        ledger = float(np.nanmean(list(self.midpoint.per_label(reading.predictions, reading.truth).values())))
        name = Artefact.stem(path)
        return Ledger75Row(
            name=name,
            ledger=ledger,
            ledger75=self.macro(reading),
            epoch=reading.epoch,
            studies=len(reading.studies),
            between_cut=self.between(reading),
            population=self.population(reading),
            void=name in self.void,
        )

    def population(self, reading: HoldoutReading) -> str:
        """A digest of the studies AND the truth they were scored against, in one canonical study order.

        A population is a QUESTION, not a study list: two members answer the same one only when the same
        studies are graded by the same table. An arm that overrides `--labels` without `--eval-labels`
        bakes its own table into its holdout truth, and its macro is then a reading of a different ruler
        that a flat rank prints in the same column — `recal_holdannot_s2026` reads 0.9004 against
        `report_labels_v3.csv` and 0.8888 against the corpus table (knee-g2vm9). Digesting the truth puts
        such a member in a population of its own, which is exactly the case `table` already refuses to
        interleave and `--only` already refuses to span.
        """
        order = np.argsort(np.asarray([str(study) for study in reading.studies]))
        studies = "|".join(str(reading.studies[index]) for index in order)
        digest = hashlib.sha1(studies.encode("utf-8"), usedforsecurity=False)
        digest.update(np.round(np.asarray(reading.truth, dtype=np.float64)[order], 6).tobytes())
        return digest.hexdigest()[:8]

    def between(self, reading: HoldoutReading) -> int:
        """How many truth cells sit in `[MIDPOINT, RECUT)` — the only cells the two rulers disagree about.

        Counted over the WHOLE truth array rather than per column, because the pair of macros this row
        carries is a whole-array statistic; the per-column version belongs to whoever reads `per_label`.
        """
        truth = np.asarray(reading.truth, dtype=np.float64)
        return int(np.count_nonzero((truth >= MIDPOINT) & (truth < self.recut.cut)))

    def table(self, paths: list[Path]) -> list[Ledger75Row]:
        """Every member that left a record, grouped by population and best LEDGER75 first inside each.

        A member whose record is missing is SKIPPED rather than raised on, because the sweep's whole
        purpose is to price everything on disk and a dark arm sitting in the same directory must not
        take the other two hundred down with it.

        The LARGEST population leads, and every other one follows whole rather than interleaved. A flat
        sort put `foreign_oof_corpus` (4406 studies) third and three `oof_gen2` fold members (849-890,
        each holding out a different quarter) inside the top sixteen, where a reader takes a rank as a
        seat ordering. Ordering ACROSS populations is the thing that cannot be done, so it is not done.
        """
        rows = []
        for path in paths:
            try:
                rows.append(self.row(path))
            except FileNotFoundError:
                logger.debug("%s has no holdout record", path)
        leading = Counter(row.population for row in rows).most_common(1)
        main = leading[0][0] if leading else ""
        return sorted(rows, key=lambda row: (row.population != main, row.population, -row.ledger75))

    @staticmethod
    def lines(rows: list[Ledger75Row]) -> list[str]:
        """The table: a header, one divider per population so no rank is read across it, and the count."""
        lines = [f"{'member':40s} {'LEDGER':>8s} {'LEDGER75':>9s} {'move':>8s} {'epoch':>6s} {'rows':>6s}"]
        seen = ""
        for row in rows:
            if row.population != seen:
                seen = row.population
                lines.append(f"-- population {seen}, {row.studies} studies - ranks below do NOT cross this line --")
            lines.append(row.line())
        lines.append(f"{len(rows)} member(s) priced under both rulers")
        return lines

    @staticmethod
    def priced(rows: list[Ledger75Row], holdouts: Path) -> str:
        """Price every non-VOID row's NaN-holdout checkpoint; a VOID number is not a held-out reading."""
        written = [row.name for row in rows if not row.void and row.price(holdouts / f"{row.name}.pt")]
        return f"priced {len(written)} checkpoint(s) that carried no holdout: {', '.join(written) or 'none'}"

    @staticmethod
    def refusal(rows: list[Ledger75Row], declines: Path, *, compared: bool) -> str:
        """Why the table may not be quoted, or "" when it may.

        A VOID row refuses always. A span of populations refuses only when `compared`: a default sweep
        across populations is a survey the dividers make honest, but `--only a b` asks for a comparison,
        and across two study sets there is none to hand back.
        """
        voided = [row.name for row in rows if row.void]
        if voided:
            return f"{len(voided)} row(s) VOID by {declines} and unrankable: {', '.join(voided)}"
        asked = {(row.population, row.studies) for row in rows if compared}
        if len(asked) <= 1:
            return ""
        spans = ", ".join(f"{name} ({studies} studies)" for name, studies in sorted(asked))
        return (
            f"--only spans {len(asked)} populations - a population is the studies AND the table that graded "
            f"them, so equal counts still answer different questions: {spans}"
        )


RULINGS: dict[Ruler, Ruling] = {
    Ruler.MIDPOINT: Ruling("offline", LabelScores),
    Ruler.LEDGER75: Ruling("offline75", lambda: LedgerSeventyFive().scorer()),
}
"""One field per ruler, so a row can never hold a number whose cut its column name does not state."""


def main() -> None:
    """Print every member on disk under the midpoint macro and LEDGER75, and the move between them."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--holdouts", type=Path, required=True, help="directory of .holdout.npz records")
    command.parser.add_argument("--only", nargs="+", default=[], help="member stems, instead of the whole directory")
    command.parser.add_argument("--declines", type=Path, default=DECLINES, help="what deliberately does not ship")
    command.parser.add_argument(
        "--price",
        action="store_true",
        help="write LEDGER into any priced checkpoint whose own holdout is NaN (a weight soup's is)",
    )
    arguments = command.parsed()

    # The DEFAULT sweep goes through the pool and the explicit ask does not, which is `MemberPool.names`'s
    # own rule: a named member is a caller pricing a specific record, void or not. What that left open is
    # that the ranking then carries the void row unlabelled, so the marker below is on the row and the
    # refusal is at the end — the number stays printable, the table stops being quotable in silence.
    pool = MemberPool.at(arguments.holdouts, arguments.declines)
    paths = [arguments.holdouts / f"{name}{SUFFIX}" for name in pool.names(arguments.only)]
    rows = LedgerSeventyFive(declines=pool.declines).table(paths)

    lines = LedgerSeventyFive.lines(rows)
    if arguments.price:
        lines.append(LedgerSeventyFive.priced(rows, arguments.holdouts))
    command.report(lines)

    refusal = LedgerSeventyFive.refusal(rows, arguments.declines, compared=bool(arguments.only))
    if refusal:
        raise SystemExit(refusal)


if __name__ == "__main__":
    main()
