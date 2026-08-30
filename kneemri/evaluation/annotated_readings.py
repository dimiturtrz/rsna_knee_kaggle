"""Our own members read over the annotated studies: `python -m kneemri.evaluation.annotated_readings`.

WHY IT EXISTS. `supervision_gap` splits the twelve findings by whether their LABELS or their MODEL is the
weaker half, and that split is currently steering which lever the remaining slots go to. It could only
ever be computed on the PUBLISHED members' out-of-fold arrays, because every member we fit weights the
annotated studies at 3.0 and scoring one there would be scoring its own training cells. So the page rests
on the assumption that our members share a stranger's deficit SHAPE, and that assumption has never been
tested.

`holdannot` was fitted to close exactly that hole — `--hold-annotations` drops the annotated rows from the
training table, so the radiologist cells are genuinely unseen — and then could not be read, because
nothing here predicts a member over a study list that is not a fold holdout. This is that step, and it is
the whole distance between a finished instrument and its reading.

THE STUDY SET IS NOT A FOLD. The annotated studies are scattered across all five, so a member's own
holdout array holds about a twelfth of them and a per-finding split read off twelve studies is a report on
the draw. This reads the member over every annotated study it has, which is legal ONLY for a run fitted
with `--hold-annotations`. Any other checkpoint is REFUSED rather than scored: a member that trained on
those cells returns a number here, not an error, and the number would be its own supervision reflected
back.

THE REFUSAL GUARDS THE TABLE ROUTE ONLY, AND THERE IS A SECOND ONE — 2026-08-26. `withheld` reads
`hold_annotations` and nothing else, so a `--calibrated-targets --hold-annotations` run passes it while
its targets still carry annotation-derived values: `Supervision.MEASURED_AGREEMENT` is a hardcoded
constant FITTED on these same 696 cells, and the hold drops the cells from the training table without
refitting it. That is not an oversight to close by refusing more checkpoints — it is a limit on how the
resulting number may be READ, and the flag is not the thing that decides it.

WHY IT IS NOT CLOSED BY A REFUSAL. `core.data.labels.level_agreement.stability` refits the constants on
disjoint halves of the annotated studies and prices the gap in its own binomial standard errors. Four of
the five levels land under 2 sigma (0.7 to 1.5) and only 0.82 reaches 2.0 — so five scalars estimated from
72-196 cells each are a population quantity, not a fit to these 58, and they cannot carry which study has
an effusion. What calibration DOES do is move targets out of report-space toward annotation-space, so a
larger gain here than on the report-label proxy is the mechanism working, not the score leaking.

SO THE NUMBER IS A CONSISTENCY CHECK AND NEVER A CONFIRMATION. `knee-bs4l`'s +0.0129 annotated against
+0.0072 proxy survives this, but a calibrated arm's annotated read is scored on the ruler its constants
were fitted against, and no stability result changes that. Quote it as agreement between two rulers that
share an origin; an INDEPENDENT annotated number needs constants refitted on a held half, which needs a
retrain, and no arm has spent one.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from core import cli
from core.paths import DataRoot
from core.wakelock import Wakelock
from kneemri.device import VisibleCards
from kneemri.evaluation.gold_rows import GoldRows
from kneemri.inference.holdout_predictions import HoldoutPredictions
from kneemri.inference.member_readings import MemberReadings
from kneemri.inference.members import Member
from kneemri.inference.prediction_cache import PredictionCache
from kneemri.inference.test_time import SpatialViews

logger = logging.getLogger(__name__)

HELD = "True"
"""What a run that withheld the annotations recorded, the options table storing every flag as a string."""


class TrainedOnAnnotationsError(Exception):
    """A checkpoint whose loss read the radiologist cells, so its score against them is not held out."""


class AnnotatedReadings:
    """Each member's probabilities for the annotated studies, over all of them rather than one fold's."""

    def __init__(self, root: DataRoot, device: torch.device, views: SpatialViews | None = None) -> None:
        self.root = root
        self.predictions = HoldoutPredictions(device, views)

    def withheld(self, checkpoint: Path) -> None:
        """Refuse a run whose loss read the annotations, naming the flag that would have made it legal.

        A refusal rather than a warning: the caller's next act is to print a per-finding table, and a
        table computed from training data is indistinguishable on the page from one computed honestly.
        """
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        options = saved.get("options", {})
        if str(options.get("hold_annotations", "")) != HELD:
            message = (
                f"{checkpoint.stem} weighted the annotated studies in its loss — score it here and it "
                f"reads its own supervision back. Refit with --hold-annotations (experiments/arms/holdannot.sh)"
            )
            raise TrainedOnAnnotationsError(message)

    def rows(self, reference: list[str], annotated: list[str]) -> NDArray[np.int64]:
        """Where the annotated studies sit in the corpus order, as the index array a slab takes.

        In corpus order rather than annotation order, because a slab reads its studies in the order its
        indices are given and every downstream join is by NAME — so the one thing that has to hold is
        that the names a caller is handed came from these same indices.
        """
        wanted = set(annotated)
        found = np.array([index for index, name in enumerate(reference) if name in wanted], dtype=np.int64)
        if found.size == 0:
            message = "none of the annotated studies are in this member's corpus — check the cache it names"
            raise ValueError(message)
        return found

    def readings(  # devtools-ignore: test-mirror
        self, checkpoints: list[Path], annotated: list[str], *, overlap: bool = False
    ) -> tuple[list[str], list[tuple[Path, NDArray[np.float64]]]]:
        """The annotated studies and each member's probabilities over them, aligned by construction.

        Grouped through `MemberReadings` so members sharing a cache pay one decode, and cached through the
        same store the curve uses, keyed on the studies actually scored — so a second reading of the same
        member over the same set costs no encoder at all.

        EACH ARRAY CARRIES THE CHECKPOINT THAT PRODUCED IT, and that is the whole reason for the pair.
        The grouping above walks members in CACHE order, not in the order they were asked for, so a caller
        that zipped the returned arrays back against its own `--checkpoints` list read every member's score
        under another member's name. `strict=True` did not catch it and could not: the lengths always
        match, and it is the IDENTITIES that were crossed. It was found on 2026-08-30 by a six-member read
        whose `--overlap` table came back byte-identical to its plain one, and it silently misreported five
        of those six rows — in the instrument that scores members against the ruler the board actually
        uses. Returning the pairing from the one place that knows it is what makes the re-derivation
        impossible rather than merely discouraged; the same defect under another name is `knee-y8y`.
        """
        for checkpoint in checkpoints:
            self.withheld(checkpoint)
        grouped = MemberReadings([Member.from_checkpoint(path) for path in checkpoints], self.root.processed())
        corpus, reference = grouped.reference()
        rows = self.rows(reference, annotated)
        names = [reference[index] for index in rows]
        cache = PredictionCache(self.root.processed() / "annotated_predictions", names, self.predictions.views)
        logger.info("%d annotated studies of %d requested are in the corpus", len(names), len(annotated))
        per_member: list[tuple[Path, NDArray[np.float64]]] = []
        with Wakelock():
            for member, slab, held in grouped.slabs(rows, reference, corpus):
                window = member.window(held.slices, overlap=overlap)
                per_member.append((member.file, self.predictions.cached(member, slab, cache, window)))
                logger.info("%s: read over %d annotated studies", member.file.name, len(names))
        return names, per_member


def main() -> None:
    """Read the given members over the annotated studies and report their agreement with the radiologist."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--checkpoints", type=Path, nargs="+", required=True, help="--hold-annotations runs")
    command.parser.add_argument("--device", default=VisibleCards().device().type, choices=("cuda", "cpu"))
    command.parser.add_argument("--overlap", action="store_true", help="read overlapping windows, as the kit does")
    command.parser.add_argument("--tta", action="store_true", help="average over rotations as the curve can")
    arguments = command.parsed()

    from kneemri.evaluation.scores import LabelScores  # noqa: PLC0415 - CLI-only
    from kneemri.inference.vote import RankAverage  # noqa: PLC0415 - CLI-only

    root = command.rooted(arguments)
    identifiers, truth = GoldRows().annotated(root.raw() / "train.csv")
    views = SpatialViews() if arguments.tta else None
    reader = AnnotatedReadings(root, torch.device(arguments.device), views)
    names, per_member = reader.readings(arguments.checkpoints, identifiers, overlap=arguments.overlap)

    position = {name: index for index, name in enumerate(identifiers)}
    annotations = truth[np.array([position[name] for name in names], dtype=np.int64)]
    scores = LabelScores()
    combined = RankAverage().combine([values for _, values in per_member])
    command.report(
        [f"{len(names)} annotated studies, {len(per_member)} member(s), none of them trained on these cells"]
        + [f"{member.stem:28s} {scores.mean(values, annotations):.4f}" for member, values in per_member]
        + [f"{'combined':28s} {scores.mean(combined, annotations):.4f}"]
    )


if __name__ == "__main__":
    main()
