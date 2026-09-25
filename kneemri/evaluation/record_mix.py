"""Score ensembles of members that are already on disk, with no card and no inference pass.

`python -m kneemri.evaluation.record_mix --records a.pt b.pt c.pt`

`member_curve` answers the same question and cannot run here: it rebuilds each member's reading from
the pixel cache, which needs the corpus, minutes, and — for anything not already in the prediction
cache — the GPU that a trainer is holding. But every run since `HoldoutRecord` existed WROTE the array
that analysis reconstructs: `Fitting` persists the kept epoch's per-study probabilities beside the
checkpoint, exact rather than re-derived. Forty-odd members are sitting there, and asking which of them
ensemble well has been costing inference passes over arrays that were already written.

WHAT IT IS FOR: diversity is the only ensemble lever this project has left (six axes measured, count
plateaus at eight, per-label selection is dead, the vote is flat), and diversity is a question about
WHICH members, which means a subset search over as many trained members as possible. That search is
arithmetic. It should not be gated on a card.

COUNT PLATEAUS, IT DOES NOT SATURATE AT FOUR — this line said "four" until 2026-08-22, when the greedy
was finally run over all 70 same-split records rather than 22. Four reads 0.8703, five 0.8710, the peak
0.8717 at eight to ten, and twenty-two still 0.8711: five through twenty-two spans 0.0007, so four is
0.0014 BELOW a seventeen-member-wide plateau. The board agrees from its own side (best entry per count
0.930 / 0.931 / 0.931 / 0.932 for four through seven). Both earlier readings were true of the pool they
were taken on, which is the failure mode to watch for here: this tool's answer moves as the pool grows,
so quote the pool size with the number or the next reader inherits a stale constant.

AND THE PLATEAU MOVES WITH THE VOTE RULE, which nobody could see until `--vote` existed. Every count
number above was taken under `RankAverage` while the board entries serve `LogitAverage`. On the 91
same-ruler records, the honest curve peaks at TEN seats / 0.8735 under rank and at FIFTEEN / 0.8747
under logit, and logit's decay past its peak is gentler (0.8736 still at twenty-four, against rank's
0.8727). So "count plateaus at eight to ten" is a fact about the RANK ruler, not about ensembling —
name the rule beside the pool size, or the constant is stale in a second way.

AND THE PEAK IS NOT A PEAK. 2026-08-27, with `--largest-pool` making the whole processed root reachable:
104 records / 909 studies puts the honest peak at THIRTEEN / 0.8754 under logit, and the 42 packaged
records put it at TWELVE / 0.8755. The count moved again, exactly as the paragraph above warns — but the
useful reading is that it moved because it is NOISE. The honest curve spans 0.0030 from six seats to
twenty-four, against a paired holdout floor of 0.0062, so no count in that range is distinguishable from
any other. The board says the same from its side: six seats and eleven both score 0.941, seven and twelve
both 0.937. What actually moves across the table is the greedy BIAS (+0.0031 at six seats, +0.0010 at
twelve) — more seats buys a steadier estimate, not a better one. Do not spend a submission on this axis,
and do NOT apply the `board_transfer` slope to a count delta: that slope was fitted on member QUALITY and
does not carry across axes. See `interpretations/kneemri/2026-08-27_seat-count-is-flat-both-rulers.md`.

A FOURTH READING, AND IT CONFIRMS THE NOISE. 2026-08-30, 113 records over the members we trained (the
six foreign ones now refused by default): the honest peak is NINETEEN / 0.8763 under logit, against the
thirteen / 0.8754 above. Nine more records, six more seats, +0.0009 — a sixth of the paired floor, and
inside the 0.0030 span this table has spread across four pools. The count is not a quantity to optimise.

READ THE NUMBER AS A SIGN, NOT AS A SCORE. Two reasons, both recorded. A training-time record was
produced by the TRAINER's reading of the holdout, so a member fitted under a different window is
scored here as it scored itself and not as the submission path would read it — comparable across
members, not identical to `member_curve`'s GPU numbers. And the holdout prices diversity too cheaply
anyway: collapsing three members into one pixel group cost 0.0059 here and 0.014 on the board, 2.4x.
What transfers from an offline member comparison is the sign.

THE RULER IS CHECKED, not assumed. Records join by study NAME and their truth vectors must agree — this
project has three label tables and has already re-run its ablations after one of them moved, so a mix
across that boundary is refused rather than averaged over.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from kneemri.evaluation.member_curve import COMBINE_SEED, MemberCurve
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.scores import LabelScores
from kneemri.inference.vote import LogitAverage, MedianLogit, RankAverage, Vote
from kneemri.runs.holdout_record import FOREIGN, RULER_TOLERANCE, HoldoutReading, HoldoutRecord

logger = logging.getLogger(__name__)

SHARED_FLOOR = 0.9
"""How much of the widest member's holdout the shared set must keep before a mix is believed.

Not a tolerance for disagreement — members on one split agree exactly, so the real values are 1.0 or a
fraction near zero. It is a floor with room for a member whose record predates a handful of studies.
"""

VOTE_RULES: dict[str, type[Vote]] = {
    "rank": RankAverage,
    "logit": LogitAverage,
    "median": MedianLogit,
}
"""The vote rules this tool can be asked about, by their CLI name.

`QualityWeightedRank` is absent because it fits its weights on the same holdout it would be scored on
here — the defect that killed per-label selection, and the reason it has never shipped. The three that
remain need nothing but the members' arrays, which is what makes them askable offline AND servable.
"""


class RecordMix:
    """The members' recorded holdout predictions, joined onto the studies all of them held out."""

    def __init__(self, records: list[Path], *, foreign: bool = False) -> None:
        self.records = records
        self.foreign = foreign
        self._readings: list[tuple[str, HoldoutReading]] | None = None
        self._dead: list[str] = []
        self._narrowed: list[str] = []

    def readings(self) -> list[tuple[str, HoldoutReading]]:
        """Each record that exists, as (member name, its reading).

        A missing record is named and skipped rather than raised on: the pool predates the record file,
        so a sweep over every checkpoint in the processed directory would otherwise be decided by its
        oldest member. What must never be silent is a record that exists and disagrees, and that is
        `shared`'s refusal.

        A path where NEITHER the checkpoint nor its record is on disk is the other fault, and it is a
        refusal. Both once printed "no holdout record, skipped", which meant a wrong root — or one
        mistyped name in nine — scored an eight-member ensemble while the caller read the number as
        nine members. The crash that exposed it only fired because EVERY path was wrong; one wrong path
        is silent, and a wrong denominator wearing a result's clothes is what this project rules out.

        A record with NO CHECKPOINT beside it is dropped and named, because it is not a member. The
        trainer writes `.holdout.npz` on every improving epoch and the `.pt` only after fitting returns,
        so a record without its checkpoint is a run that DIED — the array is real, the weights it was
        read from are gone, and nothing that wins a seat here can ever be packaged. Nine of the ninety-one
        records under the processed root are in that state (crashes, and rotated `_run1`/`_run2` runs whose
        checkpoint was overwritten), so a glob of the directory seats a tenth of its pool on runs that
        cannot ship. Dropped, not raised on, for the same reason a missing record is: a sweep must not be
        decided by its worst member. Named through `dropped`, because a pool silently one member shorter
        than the caller asked for is the wrong-denominator failure again.

        A record stamped `FOREIGN` is dropped and named unless the caller asked for one. A foreign
        record is someone else's weights served through someone else's decode, and it is decoded onto
        OUR studies under OUR label table — which is exactly what makes it mixable here and exactly what
        makes it contaminated. On 2026-08-30 a glob of every record under the processed root put the six
        of them into a seat search and returned an honest peak of 0.8994 at six seats against the 0.8754
        the ledger carries, off nothing but their presence: `raptor_v4_swa` alone reads 0.8938, and
        `foreign_overlap` had already measured 896 of our 909 holdout studies inside the labels it was
        distilled from. `foreign_overlap` cannot catch this — it takes a foreign LABEL TABLE, so it
        guards where a foreign model is PRICED and not where one is SEATED.

        Guilt by default, as `foreign_gate` already states the rule, and the pool needs no back-fill to
        obey it: `HoldoutRecord.write` has always written `source`, so all 144 records on disk carry a
        stamp and the six that are foreign say so. A caller reproducing a SHIPPED entry passes
        `foreign=True` — an entry that went to the board with a foreign member in it is a fact to be
        reproduced, not a search that seated one.

        Read once and held: `shared` and `per_member` both want them, so an uncached read decodes every
        npz three times and prints each skipped member's line three times, which reads as three
        different missing records rather than one.
        """
        if self._readings is not None:
            return self._readings
        found = []
        for path in self.records:
            record = HoldoutRecord(path)
            if not path.exists() and not record.exists():
                message = f"{path} is not on disk, and neither is {record.path} — check the root"
                raise FileNotFoundError(message)
            reading = record.read()
            if reading is None:
                logger.info("%s: no holdout record, skipped", path.name)
                continue
            if reading.source == FOREIGN and not self.foreign:
                logger.warning(
                    "%s: someone else's weights through someone else's decode — dropped, pass --foreign to seat it",
                    path.name,
                )
                continue
            # AFTER the foreign question, because the two refusals answer different things and the
            # order decided one of them wrongly. A missing checkpoint means a killed run for a member
            # WE fitted; a foreign member never has one of our checkpoints at all, by construction —
            # its record is the whole artefact. Asked first, this rule dropped `dinov3f0_solo` as a
            # dead run and made the foreign seat unreadable offline even with --foreign passed.
            if reading.source != FOREIGN and not path.exists():
                self._dead.append(path.name)
                logger.warning(
                    "%s: a holdout record with no checkpoint — the run never reached its end, dropped",
                    path.name,
                )
                continue
            found.append((path.name, reading))
        self._readings = found
        return found

    def dropped(self) -> list[str]:
        """The records that had no checkpoint, in the order they were given — what left the pool, and why.

        A caller that globbed a directory asked for a pool size; this is the difference between that
        number and the one the search actually ran over.
        """
        self.readings()
        return self._dead

    def narrow(self) -> list[str]:
        """Drop what cannot be mixed and keep the LARGEST agreeing pool — the bisect, done by the tool.

        WHY THIS EXISTS. `shared` and `per_member` both refuse rather than narrow, which is right when a
        caller named its members and wrong when it globbed a directory. The processed root holds smoke
        runs on eleven studies and members scored under an older label table beside ninety real records,
        so the glob this tool was written for cannot run: it raises on one offender, and the offender is
        named one at a time. Doing that by hand on 2026-08-27 dropped forty members without converging,
        because `per_member` compares every table against the FIRST record's — so a bisect that removes
        whatever it names walks toward whichever ruler happens to sort first, not toward the majority.

        Both refusals stay exactly as they are. This is opt-in, and it reports every drop: the pool that
        remains IS the denominator, and a search silently run over a third of the members it was handed
        is the wrong-denominator failure this file already refuses twice.
        """
        readings = self.readings()
        if not readings:
            message = (
                f"none of the {len(self.records)} named records carries a holdout array — a whole-corpus "
                f"member (--fold -1) writes no .holdout.npz, so it has nothing to be narrowed against"
            )
            raise ValueError(message)
        kept, off_split = self._on_widest_split(readings)
        self._readings = kept
        kept, off_ruler = self._largest_ruler_group(kept)
        self._readings = kept
        self._narrowed = [*off_split, *off_ruler]
        return self._narrowed

    @staticmethod
    def _on_widest_split(
        readings: list[tuple[str, HoldoutReading]],
    ) -> tuple[list[tuple[str, HoldoutReading]], list[str]]:
        """The members holding out a full split, and the names of those that hold out a stub of one.

        Filtered on the member's OWN width rather than on the intersection it causes, because that is
        the quantity `shared` refuses on, and a smoke run is short in exactly that way: eleven studies
        against 909.
        """
        widest = max(len(reading.studies) for _, reading in readings)
        kept = [(name, reading) for name, reading in readings if len(reading.studies) >= SHARED_FLOOR * widest]
        narrow = [name for name, reading in readings if len(reading.studies) < SHARED_FLOOR * widest]
        return kept, narrow

    def _largest_ruler_group(
        self, readings: list[tuple[str, HoldoutReading]]
    ) -> tuple[list[tuple[str, HoldoutReading]], list[str]]:
        """The biggest set of members whose truth agrees, and the names of everyone outside it.

        Grouped under the same `allclose` that `per_member` refuses on, so a pool this returns is one
        that `per_member` accepts — a stricter test here (exact bytes) would split a group the mix would
        have taken, and a looser one would hand it a pool it then rejects.
        """
        studies = self.shared()
        groups: list[tuple[Float[NDArray[np.float32], "shared labels"], list[tuple[str, HoldoutReading]]]] = []
        for name, reading in readings:
            _, theirs = self._rows(reading, studies)
            for truth, members in groups:
                if np.allclose(truth, theirs, atol=RULER_TOLERANCE, equal_nan=True):
                    members.append((name, reading))
                    break
            else:
                groups.append((theirs, [(name, reading)]))
        largest = max(groups, key=lambda group: len(group[1]))[1]
        seated = {name for name, _ in largest}
        return largest, [name for name, _ in readings if name not in seated]

    def narrowed(self) -> list[str]:
        """What `narrow` dropped, or nothing when it was never called — the pool's missing denominator."""
        return self._narrowed

    def shared(self) -> list[str]:
        """The studies every member held out, in the first member's order.

        Ordered by one member rather than sorted, so the truth vector below is indexed the same way the
        first record's own numbers were — and an empty intersection is an error, not an empty table:
        members from different folds produce exactly that, and their mix would otherwise report a mean
        over nothing.

        A SMALL intersection is refused for the same reason an empty one is. A smoke run holds out
        eleven studies; sweeping the processed directory put four of them beside twenty-three real
        members, and the search silently reported a greedy ordering over those eleven — a number that
        looked like every other number this tool prints. The denominator has to be checked, not
        assumed, so a member that costs the shared set most of its studies is named and the mix is
        refused rather than quietly reported over the survivors.
        """
        readings = self.readings()
        if not readings:
            message = "no member carries a holdout record"
            raise ValueError(message)
        common = set.intersection(*[set(reading.studies) for _, reading in readings])
        studies = [name for name in readings[0][1].studies if name in common]
        if not studies:
            message = f"no study is held out by all {len(readings)} members — different folds or splits"
            raise ValueError(message)
        widest = max(len(reading.studies) for _, reading in readings)
        if len(studies) < SHARED_FLOOR * widest:
            narrow = sorted(name for name, reading in readings if len(reading.studies) < widest)
            message = (
                f"{len(studies)} shared studies against {widest} in the widest member — "
                f"{', '.join(narrow)} hold out a different split (smoke runs and other folds do this)"
            )
            raise ValueError(message)
        return studies

    def per_member(self, vote: Vote | None = None) -> Pool:
        """The members' names, their aligned predictions, and the one truth they are all scored against.

        Truth is checked against the first member's for every other member, which is the same refusal
        `HoldoutReading.aligned` makes pairwise: the arithmetic of a mix succeeds across two label
        tables and answers a question nobody asked.

        `vote` defaults to the rank-average every number in this project's ledger was taken under, so an
        old reading stays comparable. Pass another rule to ask the offline ruler about the VOTE axis —
        which it could not be asked at all until 2026-08-27, while the shipped champion has served
        `LogitAverage` since `kneemri-vote`. The ruler and the entry disagreeing on the rule is a harness
        gap, not a finding.
        """
        studies = self.shared()
        readings = self.readings()
        names: list[str] = []
        predictions: list[Float[NDArray[np.float64], "shared labels"]] = []
        truth = self._rows(readings[0][1], studies)[1]
        for name, reading in readings:
            mine, theirs = self._rows(reading, studies)
            if not np.allclose(truth, theirs, atol=RULER_TOLERANCE, equal_nan=True):
                message = f"{name} was scored by a different label table over the {len(studies)} shared studies"
                raise ValueError(message)
            names.append(name)
            predictions.append(mine)
        return Pool(names, predictions, truth, vote or RankAverage(), studies)

    @staticmethod
    def _rows(
        reading: HoldoutReading, studies: list[str]
    ) -> tuple[Float[NDArray[np.float64], "shared labels"], Float[NDArray[np.float32], "shared labels"]]:
        """One reading's predictions and truth for the shared studies, in the shared order."""
        positions = {study: index for index, study in enumerate(reading.studies)}
        rows = np.array([positions[study] for study in studies], dtype=np.int64)
        return reading.predictions[rows], reading.truth[rows]


@dataclass(frozen=True)
class MixReading:
    """Which of the three questions an invocation is asking of one pool.

    Ordered by how much is held fixed. A roster pins the members that already ship and prices the rest
    against them; greedy pins nothing and builds an ensemble from scratch; the exhaustive reading pins
    nothing and enumerates. The roster case is checked first because naming one is unambiguous — nobody
    passes a shipped four-member roster and means `search every subset`.
    """

    pool: Pool
    roster: list[Path]

    def lines(self, curve: MemberCurve, seats: str | None, *, paired: bool, labels: bool, routing: bool) -> list[str]:
        """The reading this invocation asked for, as report lines."""
        if routing:
            # FIRST, because it is the one reading that survives a large pool. The default reading
            # contains it, but the default also enumerates every subset, so on more than a dozen members
            # the routing curve was unreachable through the only door it had.
            return curve.routing(self.pool)
        if labels:
            # BEFORE the roster branch, because a label reading of the SHIPPED mix is the common case
            # and it is named with the same `--roster` the seat scan is named with. A roster plus
            # `--labels` means "open up the entry I ship", never "price candidates against it".
            held, _ = self.pool.split(self.roster) if self.roster else (self.pool, None)
            return curve.labels(held)
        if self.roster:
            held, candidates = self.pool.split(self.roster)
            # BOTH readings, because they answer different questions and only one of them is a decision:
            # the add says whether a candidate brings anything the roster lacks, the swap says whether to
            # ship it and at whose expense. Printing only the add is what let every seat number quoted in
            # this repository be the price of a fifth member the pin forbids.
            return curve.marginal(held, candidates) + curve.swap(held, candidates)
        # A NAMED SEAT RULE rather than a `greedy` flag, because there are now two forward selections and
        # they are the same KIND of reading: build an ensemble from nothing, one member at a time,
        # differing only in what decides the next seat. `greedy` asks the holdout mean, `diverse` asks the
        # rank-space distance and never opens the labels. Two booleans would have made `greedy=True,
        # diverse=True` expressible and meaningless.
        if seats:
            return {"greedy": curve.greedy, "diverse": curve.diverse, "honest": curve.honest}[seats](self.pool)
        return curve.logged(self.pool, paired=paired)


def main() -> None:
    """Score every subset of the given runs from their recorded holdout predictions, with no card."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--records", type=Path, nargs="+", required=True, help="runs sharing ONE split")
    command.parser.add_argument("--paired", action="store_true", help="per-label deltas against the FIRST run")
    command.parser.add_argument(
        "--roster",
        type=Path,
        nargs="+",
        default=(),
        help="hold these members fixed and report what each OTHER record would add — the seat number",
    )
    command.parser.add_argument(
        "--seats",
        choices=("greedy", "diverse", "honest"),
        help="forward selection instead of every subset — the only tractable reading past a dozen members. "
        "`greedy` takes the member that lifts the holdout mean most; `diverse` takes the member LEAST "
        "correlated with anything already seated and never opens the labels. The comparison worth reading "
        "is the two PEAKS: whether an ensemble built for spread beats the one the quality ruler builds. "
        "`honest` is `greedy` refitted on half the studies and scored on the other half, both directions: "
        "the ONLY one of the three whose number may be quoted, because the other two choose and score on "
        "the same studies and report their own overfitting as a result.",
    )
    command.parser.add_argument(
        "--labels",
        action="store_true",
        help="the mix's own per-label AUCs, worst first, each priced in units of the final score",
    )
    command.parser.add_argument(
        "--routing",
        action="store_true",
        help="per-label member selection at every k — whether any member is a SPECIALIST rather than "
        "generally good. The default reading already prints this curve, but it enumerates every subset "
        "first, so this flag is the only way to ask a pool of more than a dozen. Read the split-half "
        "column against its own last row, which is the flat rank-average under the same estimator.",
    )
    command.parser.add_argument(
        "--vote",
        choices=tuple(VOTE_RULES),
        default="rank",
        help="the rule the mix votes under. `rank` is what every ledger number was taken under; `logit` "
        "is what the shipped champion actually serves; `median` is the robust centre of the same space. "
        "Reading the same roster under two of them is the only way to price the vote axis without a slot.",
    )
    command.parser.add_argument(
        "--largest-pool",
        action="store_true",
        help="instead of refusing on the first member that cannot be mixed, keep the largest pool that "
        "shares a split AND a label table, and name everything dropped. This is what makes a glob of the "
        "processed root runnable: smoke runs and members scored under an older table are removed as a "
        "group rather than one refusal at a time. The pool that survives is the denominator — read it "
        "off the log line, not off the number of paths passed in.",
    )
    command.parser.add_argument(
        "--foreign",
        action="store_true",
        help="seat records stamped `foreign` — someone else's weights through someone else's decode. Off "
        "by default: they are decoded onto OUR studies, which makes them mixable AND contaminated, and a "
        "glob of the processed root that seated the six of them read 0.8994 at six honest seats against "
        "0.8754 over the members we trained. Ask for this only to reproduce an entry that shipped one.",
    )
    command.parser.add_argument("--combine-seed", type=int, default=COMBINE_SEED)
    arguments = command.parsed()
    scores = LabelScores()
    roster = list(arguments.roster)
    everything = list(dict.fromkeys([*arguments.records, *roster]))
    mix = RecordMix(everything, foreign=arguments.foreign)
    if arguments.largest_pool and (narrowed := mix.narrow()):
        logger.warning("not mixable with the majority, dropped: %s", " ".join(narrowed))
    mixed = mix.per_member(VOTE_RULES[arguments.vote]())
    logger.info(
        "%d of %d records became members, over %d shared studies, voting %s",
        len(mixed.names),
        len(everything),
        len(mixed.truth),
        mixed.vote.named(),
    )
    if dead := mix.dropped():
        logger.warning("dropped, no checkpoint to package: %s", " ".join(dead))
    for line in mixed.alone(scores):
        logger.info(line)
    curve = MemberCurve(scores, arguments.combine_seed)
    reading = MixReading(mixed, roster)
    for line in reading.lines(
        curve, arguments.seats, paired=arguments.paired, labels=arguments.labels, routing=arguments.routing
    ):
        logger.info(line)


if __name__ == "__main__":
    main()
