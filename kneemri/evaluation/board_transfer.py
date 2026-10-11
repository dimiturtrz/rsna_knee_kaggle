"""Does our holdout ORDER entries the way the leaderboard does? `python -m kneemri.evaluation.board_transfer`.

`ruler_transfer` (retired, last at 1812b96e) measured the report-versus-radiologist half of the same
question and said plainly what it could not reach: "It is not the local-to-LEADERBOARD compression of
knee-7d1. That needs the board." This is that half. The board has now scored nine entries whose offline
number is our own ruler on our own members, which is enough to stop quoting a compression factor fitted
to two of them.

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
constant across the record: re-read over only the strongest entries it FALLS. As FITTED ON 2026-08-29,
from +1.55 over everything to +0.99 [+0.77, +1.18] among the entries at the top; as fitted on 2026-09-08
against a board file `board_check` had since completed, from +1.05 to +0.72. Whatever that is - a real saturation,
or the attenuation you get from dividing a board gap by an offline gap barely wider than the offline
noise - it decides how far a target is. Pricing an arm against today's champion at the whole-record
median reports what a holdout point USED to be worth, and the two differ by about a third. So
`regimes` re-reads the same estimator at floors the record itself supplies, and `report` prints the
fall rather than a number a reader could quote alone. For PRICING, the local cell is the right one
even if the saturation is only attenuation: an arm is priced from an OBSERVED offline reading, which
carries exactly the noise that attenuates the estimate.

EVERY NUMBER IN THIS DOCSTRING IS HISTORY, AND QUOTING ONE IS THE BUG. The +1.55 above was fitted on
2026-08-29 and had been carried off into a pre-registration, a kernel docstring and a board row by
2026-09-03, where it predicted 0.938-0.939 for `holdfive_v1`; the entry scored 0.936 and was written
down as "the offline gain does not survive to the board". It survived exactly. Holdfive's pool sat in a
band whose fitted slope was +0.71, which predicts 0.9366. A verdict was inverted by a constant nobody
re-read (knee-dn4t9, the THIRD time this constant was quoted from prose).

So: run the tool with `--near <the offline score you are pricing against>` and quote the cell it prints,
with its pair count. Never a scalar from any prose, this docstring included. `PRICEABLE` below makes a
thin cell say so in its own line, because annotating the number somewhere else is what the previous two
corrections did and neither held.

THE NUMBERS ABOVE MOVED ON 2026-08-29 AND THE OLD ONES WERE ONE CONTAMINATED ROW. `raptor_solo` sat on
the ensemble scope carrying offline 0.8938 against board 0.923, and `kneemri.evaluation.foreign_overlap`
had already shown 896 of the 909 studies that number was read on to be inside raptor's published
training row set: 98.6% in-sample. Voiding that one offline (the board number is honest and stays) is
the whole difference between these two readings:

    with it     +1.30 whole-record, top regime -0.41 [-0.44, +0.70], 15/149 resolved pairs inverted
    without it  +1.55 whole-record, top regime +0.99 [+0.77, +1.18],  0/127 resolved pairs inverted

So the "the ratio goes NEGATIVE at the top" reading was an artefact - eleven of those fifteen
inversions were that single entry's in-sample offline being disagreed with by an honest board - and
the remaining four are all UNDER the 0.0053 floor, which is to say the holdout never expressed a
preference for the board to reverse. **This ruler has never once mis-ordered a pair it resolved.** The
lesson is not about raptor: a guard whose default is guilt only guards what actually calls it, and
nothing in this fitter asked whether an offline number was honest before dividing by it.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from random import Random
from statistics import median

from core import cli
from kneemri.evaluation.ledger75 import Ruler
from kneemri.evaluation.score_scale import RAW, SCALES, Scale

logger = logging.getLogger(__name__)

RECORD = Path("comp/board.jsonl")
"""The machine-readable half of `comp/SUBMISSIONS.md`, which holds the same numbers as prose."""

DELTA_FLOOR = 0.002
"""An offline gap smaller than this contributes no ratio, the same guard `ruler_transfer` applied.

Dividing a board delta by a holdout delta that has nearly vanished reports the denominator rather than
the transfer, and two of these entries sit 0.0002 apart.
"""

DEFAULT_RULER = Ruler.MIDPOINT
"""The midpoint-cut holdout macro every constant in this module was fitted on.

A reading prices against the recorded row FIELD its ruler names (`Ruler.column`), so a switch cannot
silently re-derive.

`ledger75` is D25's recut (ten columns at 0.75, Medial Meniscus and Baker's at the midpoint), which D27
measured as cutting board inversions 21 -> 5 of ~530 pairs. It is NOT the default and must not become one
by convenience: every compression constant this project quotes was fitted on the midpoint field, and only
rows that carry `offline75` can be read under it — a row without one drops out rather than falling back,
because a mixed ratio is arithmetic across two questions.
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


SPREAD_DRAWS = 2000
SPREAD_SEED = 2026
SPREAD_TAILS = 0.025
"""A percentile interval on the regime-local median, by resampling the ratios that produced it.

Why the local cell needs one and the whole-record median does not: the top regime is always the
thinnest cell, and a median of a handful of pairs quoted bare is exactly the failure this module was
written against — a constant repeated from prose while the instrument was saying "direction". The
interval is what stops the local number becoming the next 1.72. Seeded, because a gate that re-draws
its own answer is not a gate.
"""


PRICEABLE = 5
"""The fewest resolved pairs a cell may hold and still be quoted as a PRICE rather than a direction.

Not a statistical threshold — there is no honest one for a median of dependent pairs — but a legibility
gate, and it exists because the failure it blocks has now happened three times (knee-dn4t9). Every
recurrence had the same shape: a cell was printed as `median +X.XX [low, high]`, the brackets read as a
fitted interval, and the scalar walked off into a pre-registration, a kernel docstring and a board row
with no memory of how many pairs made it. On 2026-09-04 the top cell held TWO pairs and still printed an
interval, because `spread` bootstraps whatever it is handed and resampling two numbers cannot report
less confidence than two numbers contain.

Five is where a bootstrap of dependent pairs stops being a restatement of its own input. Below it the
line says so in words, in the same place the number is, because an annotation somewhere else is what
the last three corrections tried and none of them held.
"""

BLUR_MEMBERS = 4
"""The member count above which an entry is shaped like the ones we actually ship.

Every entry submitted since 2026-08-19 carries at least this many seats, and the one-member rows are
the whole of the disagreement in every board-scatter reading taken so far. So the blur is reported
twice — over everything, and over this population — rather than fitted once on a mixture of two.
"""

PLURAL = {False: "", True: "s"}
UNRESOLVED_MARK = {True: "", False: " (UNRESOLVED, under the floor)"}
"""What an inverted pair's line carries when the holdout could not separate it, keyed on `resolved`."""

THIN = " — DIRECTION ONLY, too few pairs to price an arm with"
"""What a cell under `PRICEABLE` says about itself, carried beside its own number and never elsewhere."""


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


MEMBER_MEAN = "member_mean"
"""The mean of an entry's members rather than the ensemble they formed — comparable only to itself."""

EXCLUDED = "excluded"
"""A submission this record deliberately does not price. `board_transfer` reports per-scope drops, so
an excluded row is counted out loud rather than missing."""

SCOPES = frozenset({ENSEMBLE, MEMBER_MEAN, EXCLUDED})
"""The CLOSED vocabulary of the `scope` field, owned here because this module owns what it means.

`board_check` enforces it; see `board_check.SCOPES` for the failure that made a closed set necessary
(a value in neither vocabulary is green in both readers and silently absent from the fit)."""


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
    scale: Scale = RAW

    @property
    def offline_delta(self) -> float:
        """How much the holdout preferred `better`, in raw AUC. Positive by construction."""
        return self.better.offline - self.worse.offline

    @property
    def board_delta(self) -> float:
        """What the board paid for that preference, in raw AUC. NEGATIVE means the ruler mis-ordered."""
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
        """Board delta per unit of holdout delta ON THIS PAIR'S SCALE — knee-7d1's compression factor.

        The scale enters HERE and nowhere else, which is the whole design. `offline_delta`,
        `board_delta`, `resolved` and `inverted` stay in raw AUC, so `DELTA_FLOOR` and `RESOLVED` keep
        meaning what they were measured to mean and the pair set is IDENTICAL between two readings.
        A scale change then moves the ratios and nothing else, which is what makes the two comparable
        — the alternative, re-deriving the floors per scale, compares two different populations and
        calls the difference a finding.
        """
        return self.scale.delta(self.better.board, self.worse.board) / self.scale.delta(
            self.better.offline, self.worse.offline
        )


class BoardTransfer:
    """The leaderboard read against the holdout that chose what to submit.

    Every recipe decision in this project is made on a 58-study proxy and only five submissions a day
    can check one. knee-7d1 recorded the conversion as "appears to compress ~2x" from two entries, with
    the limits stated; nine entries now carry both numbers and the ledger holds them as prose, which is
    the state this project keeps paying for - the knowledge exists and does not reach the place that
    needs it.
    """

    def __init__(
        self, record: Path = RECORD, scope: str = ENSEMBLE, scale: Scale = RAW, ruler: Ruler = DEFAULT_RULER
    ) -> None:
        self.record = record
        self.scope = scope
        self.scale = scale
        self.ruler = ruler

    def _rows(self) -> list[dict[str, object]]:
        """Every recorded line, the notes included, exactly as the file holds them."""
        return [json.loads(line) for line in self.record.read_text(encoding="utf-8").splitlines() if line]

    @staticmethod
    def _carries_both(row: dict[str, object], ruler: Ruler = DEFAULT_RULER) -> bool:
        """Whether a row has BOTH numbers, which is what makes it an entry rather than a record.

        A row may legitimately carry neither: an arm fired to answer a question the holdout cannot ask
        has no offline number, and an entry that blanked has no board one. Both are worth recording and
        neither can enter a pair — before this the reader called `float(str(None))` on them and died,
        which took the whole reading down from 2026-08-17 onward rather than dropping three rows.
        """
        return row.get(ruler.column) is not None and row.get("board") is not None

    def entries(self) -> list[BoardEntry]:
        """Every entry carrying both numbers ON THIS READING'S SCOPE, weakest holdout first."""
        found = [
            BoardEntry(
                id=str(row["id"]),
                offline=float(str(row[self.ruler.column])),
                board=float(str(row["board"])),
                members=int(str(row["members"])),
            )
            for row in self._rows()
            if row.get("scope") == self.scope and self._carries_both(row, self.ruler)
        ]
        return sorted(found, key=lambda entry: entry.offline)

    def unmeasured(self) -> list[str]:
        """Rows on this scope that carry only one of the two numbers, named so the drop is never silent."""
        return [
            str(row["id"])
            for row in self._rows()
            if row.get("scope") == self.scope and "id" in row and not self._carries_both(row, self.ruler)
        ]

    def dropped(self) -> dict[str, int]:
        """How many entries each non-comparable scope cost, so the exclusion is never silent."""
        scopes = [str(row["scope"]) for row in self._rows() if "scope" in row and row["scope"] != self.scope]
        return {scope: scopes.count(scope) for scope in sorted(set(scopes))}

    def _combinations(self) -> list[BoardPair]:
        """Every pair of entries, unfiltered — the population `pairs` and `tied` partition between them."""
        found = self.entries()
        return [
            BoardPair(better=better, worse=worse, scale=self.scale)
            for index, worse in enumerate(found)
            for better in found[index + 1 :]
        ]

    def pairs(self) -> list[BoardPair]:
        """Every pair of entries the holdout can order, better-by-the-holdout first."""
        return [pair for pair in self._combinations() if pair.offline_delta >= DELTA_FLOOR]

    def tied(self) -> list[BoardPair]:
        """Every pair the holdout could NOT separate — the population that measures the BOARD.

        The complement of `pairs`, and the only cell in this record where the offline axis is held
        still. Two entries the ruler calls equal should score equally if the board were noiseless, so
        whatever board gap remains between them is the board's own repeatability rather than a transfer.
        """
        return [pair for pair in self._combinations() if pair.offline_delta < DELTA_FLOOR]

    def blur(self, least_members: int = 1) -> tuple[int, float]:
        """Pairs the holdout called equal, counted, and their median absolute BOARD gap.

        The board's repeatability at our own scale, measured on this record instead of quoted — which is
        the whole reason it lives here. It arrived on 2026-09-08 as a regression's residual sd (0.0076
        over 41 rows, halving to 0.0039 when ONE early one-member row was dropped), a shape this module
        declines in its own docstring: a least-squares line through unmatched pairs reports its extremes,
        and a constant whose value turns on which row is deleted is the next thing to be quoted from
        prose. Tied pairs need no line, no row deleted and no slope — they hold the offline axis still by
        construction.

        `least_members` restricts to entries of at least that many members, because a solo and an
        eight-seat stack are not the same instrument and every entry we now ship is the second kind.
        """
        gaps = [
            abs(pair.board_delta)
            for pair in self.tied()
            if pair.better.members >= least_members and pair.worse.members >= least_members
        ]
        return len(gaps), median(gaps) if gaps else 0.0

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
        ratios = self.ratios_above(floor)
        return len(ratios), median(ratios) if ratios else 0.0

    def ratios_above(self, floor: float) -> list[float]:
        """Every resolved pair's ratio where BOTH entries sit at or above `floor`."""
        return [pair.ratio() for pair in self.pairs() if pair.resolved and pair.worse.offline >= floor]

    def spread(self, ratios: list[float]) -> tuple[float, float]:
        """A percentile interval on the median of `ratios`, by seeded bootstrap.

        Returns the median twice when there is nothing to resample, so a caller never has to branch on
        emptiness to print a line: a one-pair cell honestly reports an interval of zero width, which
        reads as "one pair" rather than as a precision it does not have.
        """
        if len(ratios) < 2:  # noqa: PLR2004 - one draw resamples to itself; the interval is the point
            only = median(ratios) if ratios else 0.0
            return only, only
        random = Random(SPREAD_SEED)
        drawn = sorted(median(random.choices(ratios, k=len(ratios))) for _ in range(SPREAD_DRAWS))
        low = drawn[int(SPREAD_TAILS * SPREAD_DRAWS)]
        high = drawn[min(int((1.0 - SPREAD_TAILS) * SPREAD_DRAWS), SPREAD_DRAWS - 1)]
        return low, high

    def regimes(self) -> list[tuple[float, int, float, tuple[float, float]]]:
        """The compression re-read at each quantile floor: (floor, pairs, median, interval)."""
        found = []
        for fraction in REGIME_QUANTILES:
            floor = self.floor_at(fraction)
            count, value = self.compression_above(floor)
            found.append((floor, count, value, self.spread(self.ratios_above(floor))))
        return found

    @staticmethod
    def priced(count: int, value: float, spread: tuple[float, float]) -> str:
        """One cell as a line fragment, saying in place whether it may be quoted as a price.

        Here rather than at the two call sites that build cells, because the whole defect this guards is
        a number reaching a reader without the count that qualifies it — and two formatters is two
        chances for one of them to drop the qualification.
        """
        low, high = spread
        thin = THIN if count < PRICEABLE else ""
        return f"{count} resolved pairs, median {value:+.2f} [{low:+.2f}, {high:+.2f}]{thin}"

    def required_offline(
        self, target_board: float, from_board: float, from_offline: float, floor: float | None = None
    ) -> tuple[int, float]:
        """The holdout score an entry must reach for the board to pay `target_board`, and its cell's count.

        The compression is re-read among the entries at or above `floor`, defaulting to `from_offline`,
        because pricing a climb with the whole-record median prices it in a regime we have left. The
        floor is separable for the case that keeps arising and refuses itself otherwise: a climb FROM
        THE BEST ENTRY has no pair above it, so the regime it must be priced in is the one just below —
        naming that floor is a stated assumption, and defaulting to it silently would hide one.

        The arithmetic runs ON THE SCALE
        this reading was built with and comes back as AUC, which is the step that keeps being done by
        hand from a printed constant — three times wrongly (knee-dn4t9), and once badly enough to demand
        an 8-hour arm for a gap that a second scale prices at a third of the size.
        """
        count, ratio = self.compression_above(from_offline if floor is None else floor)
        if ratio == 0.0:
            return count, from_offline
        climb = self.scale.delta(target_board, from_board) / ratio
        return count, self.scale.advance(from_offline, climb)

    def report(self, near: float | None = None) -> str:
        """What the holdout is worth as a selector, in the two numbers that decide a submission.

        `near` is the offline score an arm would be priced AGAINST, and asking for it reads the ratio
        among the entries that are already there rather than across the whole record.
        """
        pairs = self.pairs()
        if not pairs:
            return f"{self.record} records no entry carrying both numbers"
        sections = (self._header(pairs), self._blur_lines(), self._asked(near), self._inversion_lines(pairs))
        return "\n".join(line for section in sections for line in section)

    def _header(self, pairs: list[BoardPair]) -> list[str]:
        """The denominator, the whole-record compression, and the per-floor regimes beneath it."""
        excluded = ", ".join(f"{count} {scope}" for scope, count in self.dropped().items())
        return [
            f"{len(self.entries())} entries on the {self.scope} scope read against `{self.ruler.column}`, "
            f"{len(pairs)} orderable pairs "
            f"(excluded: {excluded or 'none'}; "
            f"one-sided: {', '.join(self.unmeasured()) or 'none'})",
            f"compression: median {self.compression():+.2f} board points per holdout point "
            f"over the WHOLE record — the wrong number to price an arm with, see below",
        ] + [
            f"  among entries at or above {floor:.4f}: {self.priced(count, value, spread)}"
            for floor, count, value, spread in self.regimes()
        ]

    def _blur_lines(self) -> list[str]:
        """The board delta the holdout could not separate, over any entry and over full-sized ones."""
        return [
            f"board blur over the {count} pairs the holdout could not separate "
            f"(>= {least} member{PLURAL[least > 1]}): median |board| {gap:.4f} — a delta "
            f"smaller than this is a move the board cannot certify"
            for least in (1, BLUR_MEMBERS)
            for count, gap in (self.blur(least),)
        ]

    def _asked(self, near: float | None) -> list[str]:
        """The PRICE line for the floor the caller asked about, or nothing when it asked about none."""
        if near is None:
            return []
        count, value = self.compression_above(near)
        cell = self.priced(count, value, self.spread(self.ratios_above(near)))
        return [f"  PRICE AN ARM AT THIS ONE: at or above {near:.4f} (asked for), {cell}"]

    def _inversion_lines(self, pairs: list[BoardPair]) -> list[str]:
        """How often the board disagreed with the holdout, then each disagreement, board-worst first."""
        resolved = [pair for pair in pairs if pair.resolved]
        inverted = self.inversions()
        return [
            f"inversions: {len(inverted)}/{len(pairs)} pairs the board ordered the other way, "
            f"{sum(pair.inverted for pair in resolved)}/{len(resolved)} of the pairs the holdout resolved "
            f"at all (a gap >= {RESOLVED})",
        ] + [
            f"  {pair.better.id} over {pair.worse.id}: holdout {pair.offline_delta:+.4f}"
            f"{UNRESOLVED_MARK[pair.resolved]}, board {pair.board_delta:+.3f}"
            for pair in sorted(inverted, key=lambda pair: pair.board_delta)
        ]


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
    command.parser.add_argument(
        "--scale",
        default="raw",
        choices=sorted(SCALES),
        help="the scale the ratios are taken on. `raw` is AUC as recorded, the scale every constant in "
        "this module was fitted on. `probit` is the normal quantile, on which a bounded rate stops "
        "compressing near its ceiling — run it whenever a regime FALL is about to be read as a fact "
        "about transfer, because a fall is what raw AUC produces near 1.0 whether or not one is there.",
    )
    command.parser.add_argument(
        "--target",
        type=float,
        default=None,
        help="a board score to price a climb TO. Needs --from-board and --from-offline: the entry the "
        "climb starts at. Prints the holdout score required, on this reading's scale.",
    )
    command.parser.add_argument("--from-board", type=float, default=None, help="the board score climbed from")
    command.parser.add_argument("--from-offline", type=float, default=None, help="its holdout score")
    command.parser.add_argument(
        "--ruler",
        default=DEFAULT_RULER.value,
        choices=sorted(ruler.value for ruler in Ruler),
        help="which recorded holdout number to price against. `midpoint` is the `offline` field every "
        "constant here was fitted on and stays the default until the champion's regime has a priceable "
        "row under the recut. `ledger75` reads `offline75` instead and drops rows that lack one.",
    )
    arguments = command.parsed()
    reading = BoardTransfer(arguments.record, arguments.scope, SCALES[arguments.scale], Ruler(arguments.ruler))
    logger.info("scale: %s", arguments.scale)
    logger.info("%s", reading.report(arguments.near))
    if arguments.target is not None and arguments.from_board is not None and arguments.from_offline is not None:
        count, required = reading.required_offline(
            arguments.target, arguments.from_board, arguments.from_offline, arguments.near
        )
        thin = THIN if count < PRICEABLE else ""
        logger.info(
            "to reach board %.4f from (%.4f board, %.4f holdout): holdout %.4f needed, %+.4f above it, "
            "off %d resolved pairs%s",
            arguments.target,
            arguments.from_board,
            arguments.from_offline,
            required,
            required - arguments.from_offline,
            count,
            thin,
        )


if __name__ == "__main__":
    main()
