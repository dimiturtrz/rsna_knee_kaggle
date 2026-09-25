"""Combine members PER LABEL, and say honestly what that is worth.

The metric is the unweighted mean of twelve independent per-label AUCs, so the combine that matches it
is per-label. What we ship is `RankAverage.combine` — one flat rank-average of every member, used for every
label. The only other thing in the repository is `member_curve.oracle`, which picks the best member per
label USING THE ANSWERS: a ceiling that prices this work and can never be shipped. Between the flat
average and the cheat there was nothing, and this module is that thing.

THE TRAP IT IS BUILT AROUND. Choosing members per label on the holdout that then reports the number is
selection optimism, and it is severe here: twelve choices made on 881 studies with as few as 62 Fracture
positives. The in-sample number will look good whether or not the strategy is real. So the deliverable
is not the combine — it is the combine WITH an estimator that fits the choice on studies it is not
scored on.

AND IT CARRIES ITS OWN CONTROL. Selecting the top `k` members per label at `k = every member` IS the
flat rank-average, so one k curve holds both the candidate and its matched baseline under the SAME
estimator. Comparing a split-half number against a whole-holdout one would compare two rulers.

No GPU FOR A MEMBER THE PREDICTION STORE HAS ALREADY SEEN: arithmetic over the arrays
`HoldoutPredictions.cached` returns. A member it has NOT seen runs its encoder, and this line said the
unconditional "No GPU" until 2026-08-29, when two of these read 4.2 GB onto a card a trainer had reserved
29,992 MiB of. Run `card_check` first and pass `--device` — the condition is cache warmth, not the tool.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.data.corpus.folds import ReportFolds
from core.data.labels.targets import Targets
from core.findings import TARGETS
from kneemri.device import VisibleCards
from kneemri.evaluation.scores import LabelScores
from kneemri.inference.holdout_predictions import HoldoutPredictions
from kneemri.inference.member_readings import MemberReadings
from kneemri.inference.members import Member
from kneemri.inference.prediction_cache import PredictionCache
from kneemri.inference.vote import RankAverage

logger = logging.getLogger(__name__)

UNDEFINED = -np.inf
"""Where a label has one class present its AUC is nan, and a nan must sort LAST rather than compare
false against every candidate and land wherever the sort happens to leave it."""


class LabelCombine:
    """Per-label member selection, and the split-half estimate of what it buys."""

    def __init__(self, scores: LabelScores) -> None:
        self.scores = scores

    def ranking(
        self, per_member: list[NDArray[np.float64]], truth: Float[np.ndarray, "studies targets"]
    ) -> dict[str, list[int]]:
        """Members ordered best-first FOR EACH LABEL, by their AUC on the given studies.

        The studies are an argument rather than the whole holdout precisely so the caller can fit this
        on one half and apply it to the other — which is the only reason the number this module produces
        means anything.
        """
        per_label = [self.scores.per_label(member, truth) for member in per_member]
        order = range(len(per_member))
        return {
            target: sorted(order, key=lambda index: self._finite(per_label[index][target]), reverse=True)
            for target in TARGETS
        }

    def applied(
        self, per_member: list[NDArray[np.float64]], ranking: dict[str, list[int]], count: int
    ) -> NDArray[np.float64]:
        """Rank-average the top `count` members of each label into one table.

        Through `RankAverage.combine` rather than a second rank rule written here: the ensemble's voting is
        one decision with one home, and a per-label variant that ranked even slightly differently would
        report a number the submission path cannot reproduce. Each distinct subset is combined once and
        read for the labels that chose it, because twelve labels select from a handful of subsets.
        """
        combined = np.empty_like(per_member[0])
        already: dict[tuple[int, ...], NDArray[np.float64]] = {}
        for column, target in enumerate(TARGETS):
            chosen = tuple(ranking[target][:count])
            if chosen not in already:
                already[chosen] = RankAverage().combine([per_member[index] for index in chosen])
            combined[:, column] = already[chosen][:, column]
        return combined

    def halves(self, studies: int, seed: int) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
        """Two (fit, report) splits of the studies — each half fits once and reports once.

        Both directions rather than one, because a single split throws away half the evidence about a
        strategy whose whole question is whether it survives being chosen elsewhere.
        """
        order = np.random.default_rng(seed).permutation(studies)
        middle = studies // 2
        first, second = order[:middle], order[middle:]
        return [(first, second), (second, first)]

    def honest(
        self,
        per_member: list[NDArray[np.float64]],
        truth: Float[np.ndarray, "studies targets"],
        count: int,
        seed: int,
    ) -> float:
        """Fit the per-label choice on one half of the studies, score it on the other. Both ways.

        THIS IS THE NUMBER. Anything measured where it was chosen prices the holdout's noise as if it
        were signal, and with twelve choices over as few as 62 positives that noise is large enough to
        manufacture a whole result.
        """
        values = []
        for fit, report in self.halves(len(truth), seed):
            ranking = self.ranking([member[fit] for member in per_member], truth[fit])
            combined = self.applied([member[report] for member in per_member], ranking, count)
            values.append(self.scores.mean(combined, truth[report]))
        return float(np.mean(values))

    def curve(
        self, per_member: list[NDArray[np.float64]], truth: Float[np.ndarray, "studies targets"], seed: int
    ) -> list[str]:
        """Every k, in-sample beside split-half — and the last row IS the flat rank-average.

        Read the split-half column only, and read it against its own last row. The in-sample column is
        printed beside it so the SIZE of the selection optimism is visible rather than assumed: if the
        two columns diverge by more than the gain, the gain is the divergence.
        """
        lines = []
        whole = self.ranking(per_member, truth)
        for count in range(1, len(per_member) + 1):
            inside = self.scores.mean(self.applied(per_member, whole, count), truth)
            control = " (= the flat rank-average)" if count == len(per_member) else ""
            lines.append(
                f"top-{count:2d} per label:  in-sample {inside:.4f}   split-half "
                f"{self.honest(per_member, truth, count, seed):.4f}{control}"
            )
        return lines

    @staticmethod
    def _finite(value: float) -> float:
        """An undefined AUC, ordered rather than left to the sort's mercy."""
        return UNDEFINED if np.isnan(value) else value


def main() -> None:
    """Read the k curve over members that already have holdout predictions.

    THIS MODULE HAD NO ENTRYPOINT FOR SIX DAYS. It was written on 2026-08-20, cited twice as prior art —
    once in the ledger and once in an interpretation — and never once produced a number, because a
    library with no `main` is a claim rather than a measurement. The ledger's own sentence, "a per-label
    roster is prior art in this repo", was true about the CODE and false about the KNOWLEDGE.

    IT COSTS NO CARD when the members are already on `member_curve`'s prediction store: the arrays are
    read back by hash, and the encoder runs only for a member the store has never seen. Pass `--refresh`
    and it will run them, which on a shared card is the expensive mistake this docstring exists to warn
    about.
    """
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--checkpoints", type=Path, nargs="+", required=True, help="members sharing ONE split")
    command.parser.add_argument("--labels", type=Path, required=True, help="the report-derived table to score on")
    command.parser.add_argument("--fold", type=int, default=0)
    command.parser.add_argument("--folds", type=int, default=5)
    command.parser.add_argument("--split-seed", type=int, default=2026)
    command.parser.add_argument("--seed", type=int, default=2026, help="the half-split, NOT the fold split")
    command.parser.add_argument("--overlap", action="store_true", help="every window start, not just disjoint ones")
    command.parser.add_argument("--refresh", action="store_true", help="recompute rather than reuse stored predictions")
    command.parser.add_argument(
        "--device",
        default=VisibleCards().device().type,
        choices=("cuda", "cpu"),
        help="where to read cache MISSES. The docstring's 'costs no card' holds only for members the "
        "store has already seen; a cold member runs its encoder, and on 2026-08-29 two of these read "
        "4.2 GB onto a card a trainer had reserved 29,992 MiB of, leaving 634 MiB free. `member_curve` "
        "has taken this flag for exactly that reason and this CLI could not say cpu at all.",
    )
    arguments = command.parsed()

    root = command.rooted(arguments)
    readings = MemberReadings([Member.from_checkpoint(path) for path in arguments.checkpoints], root.processed())

    # The split comes from the FIRST cache and is reused, so every member is scored on the same studies
    # in the same order — the same discipline `ruler_check` follows, and for the same reason: a member
    # read over a differently-ordered study list is compared against the wrong truth rows and returns a
    # number rather than an error.
    corpus, identifiers = readings.reference()
    folds = ReportFolds(arguments.folds, arguments.split_seed)
    _, holdout = folds.holdout(folds.assign(str(root.train_table()), identifiers), arguments.fold)
    truth = Targets.load(arguments.labels, root.train_table(), identifiers).scores[holdout]

    predictions = HoldoutPredictions(torch.device(arguments.device))
    store = PredictionCache(
        root.processed() / "holdout_predictions",
        [identifiers[position] for position in holdout],
        predictions.views,
        refresh=arguments.refresh,
    )

    per_member = []
    for member, slab, held in readings.slabs(holdout, identifiers, corpus):
        window = member.window(held.slices, overlap=arguments.overlap)
        per_member.append(predictions.cached(member, slab, store, window))
        logger.info("read %s", member.file.stem)

    for line in LabelCombine(LabelScores()).curve(per_member, truth, arguments.seed):
        logger.info("%s", line)


if __name__ == "__main__":
    main()
