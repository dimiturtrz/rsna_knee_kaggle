"""Does our holdout ORDER entries the way the leaderboard does? `python -m kneemri.evaluation.board_transfer`.

`ruler_transfer` measures the report-versus-radiologist half of the same question and says plainly what
it cannot reach: "It is not the local-to-LEADERBOARD compression of knee-7d1. That needs the board."
This is that half. The board has now scored nine entries whose offline number is our own ruler on our
own members, which is enough to stop quoting a compression factor fitted to two of them.

PAIRS, NOT A REGRESSION. A line through nine points would be a claim about the LEVEL of two scores that
were never meant to agree - one is measured against LLM-read report labels on a 58-study holdout, the
other against radiologist annotations on the competition's hidden set. What a ruler is FOR is ordering
candidates, so every pair of entries is asked the only question that decides a submission: the holdout
preferred this one, did the board? A pair whose two deltas disagree in SIGN is an inversion, and an
inversion is the failure mode that costs a slot. The compression factor is then the median of the
pairwise ratios, which needs no linearity and is not dragged by the two entries furthest apart.

WHAT THE PAIRS ARE NOT MATCHED ON, stated because it bounds every number here: these entries span six
days, and the kit changed underneath them - fp16 serving, the image budget, overlap TTA, a member pool
that doubled. Only the two 17:45 entries on 2026-08-16 are a same-day matched pair. So a single
pairwise ratio is a confounded read and only the DISTRIBUTION is worth quoting; and an inversion says
the ruler and the board disagreed, not that the ruler is wrong about the axis its entry was built on.

ONE MEDIAN IS THE WRONG SHAPE FOR THIS, and the report says so in its own output. The ratio is not
constant across the record: re-read over only the strongest entries it FALLS, from +1.72 over
everything to under +1.00 among the entries at the top. Whatever that is - a real saturation, or the
attenuation you get from dividing a board gap by an offline gap barely wider than the offline noise -
it decides how far a target is. Pricing an arm against today's champion at the whole-record median
reports what a holdout point USED to be worth, and the two differ by about a factor of two, which is
the difference between a goal one arm away and a goal nothing on the queue can reach. So `regimes`
re-reads the same estimator at floors the record itself supplies, and `report` prints the fall rather
than a number a reader could quote alone. For PRICING, the local cell is the right one even if the
saturation is only attenuation: an arm is priced from an OBSERVED offline reading, which carries
exactly the noise that attenuates the estimate.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from core import cli

logger = logging.getLogger(__name__)

RECORD = Path("comp/board.jsonl")
"""The machine-readable half of `comp/SUBMISSIONS.md`, which holds the same numbers as prose."""

DELTA_FLOOR = 0.002
"""An offline gap smaller than this contributes no ratio, the same guard `ruler_transfer` applies.

Dividing a board delta by a holdout delta that has nearly vanished reports the denominator rather than
the transfer, and two of these entries sit 0.0002 apart.
"""

RESOLVED = 0.0053
"""The holdout's own noise floor: below it the ruler expressed no preference to be wrong about.

This is the line that decides what an inversion MEANS, so it is not a display detail. A pair the ruler
separates by less than its own floor is two entries it called equal; the board ordering them either way
refutes nothing. A pair it separates by more is a claim, and the board reversing THAT is the failure
that costs a slot.
"""


REGIME_QUANTILES = (0.5, 0.75)
"""Where to re-read the compression, as fractions through the entries' OWN offline scores.

Not invented levels. The whole point of re-reading is that the ratio moves with the score being priced,
so the only cut points that are not a thumb on the scale are the ones the record supplies. Two are
enough to show the direction; more would only invite a reader to quote the smallest cell as a number.
"""


ENSEMBLE = "ensemble"
"""The default scope: the holdout AUC of the thing that was SUBMITTED.

Two offline numbers can only be subtracted WITHIN one scope, and mixing them is the whole difference
between a result and an artefact — a MEMBER_MEAN is lower than the ensemble it describes by however
much averaging bought, so a one-member entry at 0.8518 outranks a five-member entry at 0.8269 on paper
while the board has them the other way round, and the "inversion" is the two denominators, not the
board.

A scope is a PARAMETER rather than a filter hard-coded here, because the four member_mean entries are
comparable to EACH OTHER even though they are comparable to nothing else. Read on their own they carry
the reading this project's slot decisions turn on: the two whole-stack entries have member means 0.021
ABOVE the five-fold entry and scored 0.006 BELOW it on the board, which is the member-count effect with
member quality held against it.
"""


@dataclass(frozen=True)
class BoardEntry:
    """One submission that carries BOTH numbers: what our ruler said, and what the board said."""

    id: str
    offline: float
    board: float
    members: int


@dataclass(frozen=True)
class BoardPair:
    """Two entries and the two deltas between them, in the order the holdout ranked them."""

    better: BoardEntry
    worse: BoardEntry

    @property
    def offline_delta(self) -> float:
        """How much the holdout preferred `better`. Positive by construction."""
        return self.better.offline - self.worse.offline

    @property
    def board_delta(self) -> float:
        """What the board paid for that preference. NEGATIVE means the ruler ordered this pair wrong."""
        return self.better.board - self.worse.board

    @property
    def inverted(self) -> bool:
        """Whether the board reversed the holdout's ordering of these two entries."""
        return self.board_delta < 0

    @property
    def resolved(self) -> bool:
        """Whether the holdout separated these two by more than its own noise floor at all."""
        return self.offline_delta >= RESOLVED

    def ratio(self) -> float:
        """Board delta per unit of holdout delta — knee-7d1's compression factor, for this pair."""
        return self.board_delta / self.offline_delta


class BoardTransfer:
    """The leaderboard read against the holdout that chose what to submit.

    Every recipe decision in this project is made on a 58-study proxy and only five submissions a day
    can check one. knee-7d1 recorded the conversion as "appears to compress ~2x" from two entries, with
    the limits stated; nine entries now carry both numbers and the ledger holds them as prose, which is
    the state this project keeps paying for - the knowledge exists and does not reach the place that
    needs it.
    """

    def __init__(self, record: Path = RECORD, scope: str = ENSEMBLE) -> None:
        self.record = record
        self.scope = scope

    def _rows(self) -> list[dict[str, object]]:
        """Every recorded line, the notes included, exactly as the file holds them."""
        return [json.loads(line) for line in self.record.read_text(encoding="utf-8").splitlines() if line]

    @staticmethod
    def _carries_both(row: dict[str, object]) -> bool:
        """Whether a row has BOTH numbers, which is what makes it an entry rather than a record.

        A row may legitimately carry neither: an arm fired to answer a question the holdout cannot ask
        has no offline number, and an entry that blanked has no board one. Both are worth recording and
        neither can enter a pair — before this the reader called `float(str(None))` on them and died,
        which took the whole reading down from 2026-08-17 onward rather than dropping three rows.
        """
        return row.get("offline") is not None and row.get("board") is not None

    def entries(self) -> list[BoardEntry]:
        """Every entry carrying both numbers ON THIS READING'S SCOPE, weakest holdout first."""
        found = [
            BoardEntry(
                id=str(row["id"]),
                offline=float(str(row["offline"])),
                board=float(str(row["board"])),
                members=int(str(row["members"])),
            )
            for row in self._rows()
            if row.get("scope") == self.scope and self._carries_both(row)
        ]
        return sorted(found, key=lambda entry: entry.offline)

    def unmeasured(self) -> list[str]:
        """Rows on this scope that carry only one of the two numbers, named so the drop is never silent."""
        return [
            str(row["id"])
            for row in self._rows()
            if row.get("scope") == self.scope and "id" in row and not self._carries_both(row)
        ]

    def dropped(self) -> dict[str, int]:
        """How many entries each non-comparable scope cost, so the exclusion is never silent."""
        scopes = [str(row["scope"]) for row in self._rows() if "scope" in row and row["scope"] != self.scope]
        return {scope: scopes.count(scope) for scope in sorted(set(scopes))}

    def pairs(self) -> list[BoardPair]:
        """Every pair of entries the holdout can order, better-by-the-holdout first."""
        found = self.entries()
        return [
            BoardPair(better=better, worse=worse)
            for index, worse in enumerate(found)
            for better in found[index + 1 :]
            if better.offline - worse.offline >= DELTA_FLOOR
        ]

    def inversions(self) -> list[BoardPair]:
        """The pairs the board ordered the other way — the ones that would have cost a slot."""
        return [pair for pair in self.pairs() if pair.inverted]

    def compression(self) -> float:
        """The MEDIAN board-per-holdout ratio over the RESOLVED pairs, or 0.0 when there are none.

        The median rather than a fitted slope: the pairs are not independent (five entries make ten of
        them) and are not matched, so a least-squares line would report the two extreme entries with a
        precision the data does not have.

        Resolved pairs only, because a ratio whose denominator is under the holdout's noise floor is a
        ratio of noise: the two entries the board separates by 0.008 sit 0.0029 apart on a ruler that
        cannot see 0.0053, and dividing those gives 2.8 from a gap the ruler never claimed.
        """
        ratios = [pair.ratio() for pair in self.pairs() if pair.resolved]
        return median(ratios) if ratios else 0.0

    def floor_at(self, fraction: float) -> float:
        """The offline score `fraction` of the way up the entries, 0.0 when there are none."""
        found = self.entries()
        if not found:
            return 0.0
        return found[min(int(fraction * len(found)), len(found) - 1)].offline

    def compression_above(self, floor: float) -> tuple[int, float]:
        """Resolved pairs sitting ENTIRELY at or above `floor`, counted, and their median ratio.

        `compression` answers what a holdout point has been worth across the whole record. Pricing an
        arm asks the narrower question - what is it worth HERE, against the entry we would be beating -
        and those are different numbers as soon as the ratio stops being flat. The count is returned
        beside the median because the top cell is always the thinnest, and a median of a handful of
        pairs is a direction rather than a constant.
        """
        ratios = [pair.ratio() for pair in self.pairs() if pair.resolved and pair.worse.offline >= floor]
        return len(ratios), median(ratios) if ratios else 0.0

    def regimes(self) -> list[tuple[float, int, float]]:
        """The compression re-read at each quantile floor: (floor, resolved pairs, median ratio)."""
        found = []
        for fraction in REGIME_QUANTILES:
            floor = self.floor_at(fraction)
            count, value = self.compression_above(floor)
            found.append((floor, count, value))
        return found

    def report(self, near: float | None = None) -> str:
        """What the holdout is worth as a selector, in the two numbers that decide a submission.

        `near` is the offline score an arm would be priced AGAINST, and asking for it reads the ratio
        among the entries that are already there rather than across the whole record.
        """
        pairs = self.pairs()
        if not pairs:
            return f"{self.record} records no entry carrying both numbers"
        inverted = self.inversions()
        excluded = ", ".join(f"{count} {scope}" for scope, count in self.dropped().items())
        lines = [
            f"{len(self.entries())} entries on the {self.scope} scope, {len(pairs)} orderable pairs "
            f"(excluded: {excluded or 'none'}; "
            f"one-sided: {', '.join(self.unmeasured()) or 'none'})",
            f"compression: median {self.compression():+.2f} board points per holdout point "
            f"over the WHOLE record — the wrong number to price an arm with, see below",
        ]
        lines += [
            f"  among entries at or above {floor:.4f}: {count} resolved pairs, median {value:+.2f}"
            for floor, count, value in self.regimes()
        ]
        if near is not None:
            count, value = self.compression_above(near)
            lines.append(f"  among entries at or above {near:.4f} (asked for): {count} pairs, median {value:+.2f}")
        lines += [
            f"inversions: {len(inverted)}/{len(pairs)} pairs the board ordered the other way, "
            f"{sum(pair.inverted for pair in pairs if pair.resolved)}/"
            f"{sum(pair.resolved for pair in pairs)} of the pairs the holdout resolved at all "
            f"(a gap >= {RESOLVED})",
        ]
        lines += [
            f"  {pair.better.id} over {pair.worse.id}: holdout {pair.offline_delta:+.4f}"
            f"{'' if pair.resolved else ' (UNRESOLVED, under the floor)'}, "
            f"board {pair.board_delta:+.3f}"
            for pair in sorted(inverted, key=lambda pair: pair.board_delta)
        ]
        return "\n".join(lines)


def main() -> None:
    """Read the board against the holdout: `python -m kneemri.evaluation.board_transfer`."""
    command = cli.Command(BoardTransfer.__doc__, logger)
    command.parser.add_argument("--record", type=Path, default=RECORD, help="the scored-submission record")
    command.parser.add_argument(
        "--scope",
        default=ENSEMBLE,
        help="which denominator to read within. Entries on different scopes are not subtractable, so "
        "each scope is its own reading rather than one mixed ranking.",
    )
    command.parser.add_argument(
        "--near",
        type=float,
        default=None,
        help="the offline score an arm would be priced against. The compression is re-read among the "
        "entries at or above it, because the whole-record median describes a regime we have left.",
    )
    arguments = command.parsed()
    logger.info("%s", BoardTransfer(arguments.record, arguments.scope).report(arguments.near))


if __name__ == "__main__":
    main()
