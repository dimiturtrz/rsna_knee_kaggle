"""What several members predicted for the same studies — the input every offline reading joins on.

It lives here rather than beside `MemberCurve` because two modules speak it: `member_curve` reads a
pool, `record_mix` builds one out of holdout records, and a type both of them name is package
vocabulary rather than one producer's supplementary type. The immediate cause of the move was the
god-file gate — `member_curve.py` crossed 750 lines — but the seam it picked was the right one, which is
what a threshold is for.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from kneemri.evaluation.scores import LabelScores
from kneemri.inference.vote import RankAverage, Vote


@dataclass(frozen=True)
class Pool:
    """What several members predicted for the same studies, in one order — every reading's input.

    A type rather than three parameters threaded through nine signatures. The arrays are only meaningful
    TOGETHER: a name list reordered against a prediction list mislabels every row of every report and
    nothing fails, and a truth vector from a different fold scores the wrong studies just as quietly.
    Passed as one value, the three cannot come apart at a call site.

    It carries the operations every reading repeated by hand — combine a subset, name a subset, score a
    member alone — because `RankAverage().combine([predictions[i] for i in subset])` appeared in four
    methods and each copy was a chance to combine one list while naming another.
    """

    names: list[str]
    predictions: list[Float[np.ndarray, "studies targets"]]
    truth: Float[np.ndarray, "studies targets"]
    vote: Vote = field(default_factory=RankAverage)
    """HOW these members combine. Defaults to the flat rank-average because that is the rule every
    reading in `experiments/README.md` was taken under, and a default that silently changed it would
    reprice the whole ledger. Pass another rule to ask what the offline ruler says about the VOTE axis
    — which it could not be asked at all until 2026-08-27, while the champion has shipped
    `LogitAverage` since `kneemri-vote`. The ruler and the entry disagreeing on the rule is a harness
    gap, not a finding, and this field is what closes it."""

    def __len__(self) -> int:
        """How many members — the range every subset enumeration runs over."""
        return len(self.predictions)

    def combined(self, subset: tuple[int, ...] | list[int]) -> Float[np.ndarray, "studies targets"]:
        """These members under this pool's vote rule — the flat per-label rank-average unless told otherwise."""
        return self.vote.combine([self.predictions[index] for index in subset])

    def named(self, subset: tuple[int, ...] | list[int]) -> str:
        """The subset as the names a submission would package."""
        return " + ".join(self.names[index] for index in subset)

    def alone(self, scores: LabelScores) -> list[str]:
        """Each member's own mean score on the shared studies — the column every mix is read against."""
        return [
            f"{name} alone: {scores.mean(member, self.truth):.4f}"
            for name, member in zip(self.names, self.predictions, strict=True)
        ]

    def split(self, roster: list[Path]) -> tuple["Pool", "Pool"]:
        """The roster as a pool, and the rest as candidates against it.

        The roster comes back NAMED rather than as bare arrays. A swap has to say which incumbent it
        displaces — `calib_span for calibrated_targets` is the whole content of the line — and a list of
        arrays cannot say it. An add did not need the names, which is why they were dropped here.

        The roster is read through the SAME join as the candidates rather than scored on its own: the
        shared-study intersection and the label-table refusal are the reason this type exists, and a
        roster loaded separately would be compared to candidates over a different denominator.

        A named roster member that carries no record is an error, not a skip. Elsewhere a missing
        record means an old checkpoint and skipping it costs one row; here it silently scores a
        four-member roster as three and every delta below it is against the wrong base.
        """
        wanted = {path.name for path in roster}
        found = {name for name in self.names if name in wanted}
        if found != wanted:
            message = f"roster members with no holdout record: {', '.join(sorted(wanted - found))}"
            raise ValueError(message)
        paired = list(zip(self.names, self.predictions, strict=True))
        held = [(name, member) for name, member in paired if name in wanted]
        rest = [(name, member) for name, member in paired if name not in wanted]
        return (
            Pool([name for name, _ in held], [member for _, member in held], self.truth, self.vote),
            Pool([name for name, _ in rest], [member for _, member in rest], self.truth, self.vote),
        )

    def rows(self, studies: NDArray[np.int64]) -> "Pool":
        """This same pool over a SUBSET of the studies, members and truth restricted together.

        Restricting members and truth in one place is the point: a seat chosen on one half of the
        holdout and scored on the other is only honest if both halves index the same rows, and the two
        arrays live in different attributes here. Every caller that sliced one of them separately is a
        caller that could have scored a member against another member's truth.
        """
        return Pool(self.names, [member[studies] for member in self.predictions], self.truth[studies], self.vote)
