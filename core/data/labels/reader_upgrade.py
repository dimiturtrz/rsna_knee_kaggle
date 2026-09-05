"""Does a new reader beat the lexicon ON THE PAIRS THE LEXICON LEFT BLANK?

`python -m core.data.labels.reader_upgrade --candidate <table.csv>`

THE GATE THIS IS, AND WHY THE DENOMINATOR IS NOT ALL PAIRS. 27.3% of the current supervision is a
CONSTANT: 14,445 of 52,872 (study, finding) pairs in `report_labels_v2.csv` carry the abstention rung and
assert nothing. A model reader is worth hours of card only if it SPEAKS where the lexicon was silent and
is RIGHT when it does. On the pairs the lexicon already read, the two tables largely agree by
construction — both are reading the same sentence — so an accuracy averaged over all pairs would be
dominated by cells this file exists to leave alone, and would report a win for a reader that added
nothing.

SO THERE ARE TWO NUMBERS AND THEY ANSWER DIFFERENT QUESTIONS. On the BLANK pairs: how often does the
candidate speak, and how often is it right when it does — against the majority class on those same
cells, because a reader that says "absent" to everything is right most of the time and has told us
nothing. On the pairs the lexicon ANSWERED: does the candidate hold that ground, since a table that buys
coverage by regressing where supervision already worked is a net loss the blank-pair number cannot see.

WHAT IT CANNOT SAY. The truth here is the image annotation, which exists for 56 studies once joined —
the same small denominator every reading of the reader has. A rate on the blank cells of 56 studies is
resolved to its own bootstrap width and no finer, and the interval is over STUDIES because that is the
unit a reader's failures cluster in.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.data.labels.reader_agreement import (
    DEFAULT_TABLE,
    INTERVAL,
    MIDPOINT,
    POSITIVE,
    SEED,
    SILENT,
    ReaderAgreement,
)

logger = logging.getLogger(__name__)

DRAWS = 2000
"""Bootstrap resamples of the STUDIES, the unit a reader's failures cluster in."""

KEY = ["study", "target"]
"""The cell identity both tables are joined on — a reading is a claim about one finding in one study."""


class ReaderUpgrade:
    """A candidate label table against the lexicon's, on the cells where the lexicon abstained."""

    def __init__(self, lexicon: pd.DataFrame, candidate: pd.DataFrame) -> None:
        self.lexicon = lexicon
        self.candidate = candidate

    def blank(self) -> pd.DataFrame:
        """Candidate cells whose lexicon verdict was the abstention rung — where the headroom is.

        Joined on (study, target) rather than positionally because the two tables need not carry the same
        studies in the same order, and a silent misalignment here would score one reader's cells against
        another's truth.
        """
        silent = self.lexicon[self.lexicon["verdict"] == SILENT][KEY]
        return self.candidate.merge(silent, on=KEY, how="inner")

    def answered(self) -> pd.DataFrame:
        """Candidate cells the lexicon DID read — the ground a coverage gain must not buy with a loss."""
        spoken = self.lexicon[self.lexicon["verdict"] != SILENT][KEY]
        return self.candidate.merge(spoken, on=KEY, how="inner")

    def matched(self) -> pd.DataFrame:
        """The cells BOTH readers asserted, carrying each one's verdict against the same truth.

        The answered-pair line above is not a comparison: it scores the candidate on the subset the
        CANDIDATE chose to speak on, while the lexicon's own rate is over every cell it answered. A
        reader that abstains on the hard cells wins that line by declining them. Only the intersection
        prices the correction, and correcting is the only thing left for a reader whose blank-pair
        accuracy does not clear the majority class.
        """
        spoken = self.lexicon[self.lexicon["verdict"] != SILENT][[*KEY, "verdict", "truth"]]
        mine = self.candidate[self.candidate["verdict"] != SILENT][[*KEY, "verdict"]]
        return spoken.merge(mine, on=KEY, how="inner", suffixes=("_lexicon", "_candidate"))

    @staticmethod
    def correction(matched: pd.DataFrame) -> tuple[int, float, float]:
        """(shared cells, the lexicon's accuracy on them, the candidate's) — the same cells for both."""
        if matched.empty:
            return 0, float("nan"), float("nan")
        positive = matched["truth"] > MIDPOINT
        lexicon = float(((matched["verdict_lexicon"] == POSITIVE) == positive).mean())
        candidate = float(((matched["verdict_candidate"] == POSITIVE) == positive).mean())
        return len(matched), lexicon, candidate

    @staticmethod
    def accuracy(cells: pd.DataFrame) -> tuple[int, float, float]:
        """(cells the candidate asserted, its accuracy on them, the majority-class rate on the same cells).

        The majority rate is the honest baseline: these findings are mostly absent, so a reader that never
        says YES scores well and has added no information. The comparison is the candidate against THAT,
        never against chance.
        """
        asserted = cells[cells["verdict"] != SILENT]
        if asserted.empty:
            return 0, float("nan"), float("nan")
        positive = asserted["truth"] > MIDPOINT
        correct = float(((asserted["verdict"] == POSITIVE) == positive).mean())
        prevalence = float(positive.mean())
        return len(asserted), correct, max(prevalence, 1.0 - prevalence)

    def resampled(self, cells: pd.DataFrame) -> Float[NDArray[np.float64], " draws"]:
        """The candidate's accuracy under `DRAWS` resamples of the studies, not of the cells.

        Each study is reduced ONCE to (hits, asserted) and the draws are summed over those pairs, rather
        than re-filtering the frame two thousand times. Same estimator, and it costs a second instead of
        minutes — which is what makes the interval something to report rather than something to skip.
        """
        asserted = cells[cells["verdict"] != SILENT]
        hit = (asserted["verdict"] == POSITIVE) == (asserted["truth"] > MIDPOINT)
        per_study = asserted.assign(hit=hit).groupby("study")["hit"].agg(["sum", "count"])
        hits = per_study["sum"].to_numpy(np.float64)
        counts = per_study["count"].to_numpy(np.float64)

        generator = np.random.default_rng(SEED)
        drawn = generator.integers(0, len(hits), size=(DRAWS, len(hits)))
        return np.asarray(hits[drawn].sum(axis=1) / counts[drawn].sum(axis=1), dtype=np.float64)

    def resampled_correction(self, matched: pd.DataFrame) -> Float[NDArray[np.float64], " draws"]:
        """The candidate-minus-lexicon accuracy gap under `DRAWS` resamples of the studies.

        PAIRED: each draw resamples studies once and scores both readers on the same drawn cells, so the
        shared sampling error cancels instead of being counted twice. An unpaired interval on two rates
        measured on identical cells is wide by construction and would hide a real correction.
        """
        positive = matched["truth"] > MIDPOINT
        gap = ((matched["verdict_candidate"] == POSITIVE) == positive).astype(float) - (
            (matched["verdict_lexicon"] == POSITIVE) == positive
        ).astype(float)
        per_study = matched.assign(gap=gap).groupby("study")["gap"].agg(["sum", "count"])
        gaps = per_study["sum"].to_numpy(np.float64)
        counts = per_study["count"].to_numpy(np.float64)

        generator = np.random.default_rng(SEED)
        drawn = generator.integers(0, len(gaps), size=(DRAWS, len(gaps)))
        return np.asarray(gaps[drawn].sum(axis=1) / counts[drawn].sum(axis=1), dtype=np.float64)

    def report(self) -> list[str]:
        """Both numbers and the branch they fire, so the decision is not left to whoever reads the log."""
        blank, answered = self.blank(), self.answered()
        spoken, correct, majority = self.accuracy(blank)
        held, holding, _ = self.accuracy(answered)
        coverage = spoken / len(blank) if len(blank) else float("nan")
        lines = [
            f"{len(blank)} pairs the lexicon left blank; the candidate speaks on {spoken} ({coverage:.1%})",
            f"blank-pair accuracy {correct:.3f} against a majority-class {majority:.3f}",
            f"the lexicon's own {held} answered pairs: candidate accuracy {holding:.3f}",
        ]
        if not spoken:
            lines.append("SILENT TOO: the candidate abstains where the lexicon did. No sweep — it is the same table.")
            return lines
        low, high = np.percentile(self.resampled(blank), INTERVAL)
        lines.append(f"blank-pair accuracy interval [{low:.3f}, {high:.3f}] over studies")
        if low > majority:
            lines.append(
                "READS THE BLANKS: the candidate speaks where the lexicon abstained and beats the "
                "majority class doing it. This licenses the full sweep; it does not price the arm."
            )
        else:
            lines.append(
                "NOT DISTINGUISHABLE FROM THE MAJORITY CLASS: the candidate speaks, but its accuracy on "
                "the blank pairs is not resolved above saying 'absent' to all of them. Do NOT spend hours "
                "of card on the sweep — a table built from this would add confident noise, not supervision."
            )
        return lines + self.correction_lines()

    def correction_lines(self) -> list[str]:
        """The matched reading: on the cells both readers assert, is the candidate a CORRECTOR?

        This is the branch the blank-pair gate leaves open. A candidate can add nothing where the
        lexicon is silent and still be worth a sweep if it is right more often where the lexicon
        SPOKE — that is the whole of Wollek's parser-to-learned-labeler gap, which is a correction
        result and not a coverage one.
        """
        matched = self.matched()
        shared, lexicon, candidate = self.correction(matched)
        if not shared:
            return ["no cell is asserted by both readers; the correction lane is unreadable here"]
        low, high = np.percentile(self.resampled_correction(matched), INTERVAL)
        lines = [
            f"the {shared} cells BOTH readers assert: lexicon {lexicon:.3f}, candidate {candidate:.3f}",
            f"candidate minus lexicon {candidate - lexicon:+.3f}, paired interval [{low:+.3f}, {high:+.3f}]",
        ]
        if low > 0.0:
            lines.append(
                "CORRECTS: on the same cells, read against the same truth, the candidate is right more "
                "often than the lexicon and the paired interval clears zero. The sweep is licensed by "
                "the CORRECTION lane even when the coverage lane is dead."
            )
        else:
            lines.append(
                "DOES NOT CORRECT: the gap on the shared cells does not clear zero. With the blank-pair "
                "lane also dead this candidate is not a better reader on any denominator we can measure."
            )
        return lines


def main() -> None:
    """Score a candidate label table against the lexicon's on the annotated studies."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--candidate", type=Path, required=True, help="the new reader's table")
    command.parser.add_argument("--labels", type=Path, help="the lexicon's table; defaults to report_labels_v2.csv")
    arguments = command.parsed()

    root = command.rooted(arguments)
    agreement = ReaderAgreement()
    annotated = agreement.annotated(root)
    lexicon = agreement.cells(annotated, pd.read_csv(arguments.labels or root.processed() / "labels" / DEFAULT_TABLE))
    candidate = agreement.cells(annotated, pd.read_csv(arguments.candidate))

    command.report(ReaderUpgrade(lexicon, candidate).report())


if __name__ == "__main__":
    main()
