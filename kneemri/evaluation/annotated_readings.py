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

AND THERE IS A THIRD ROUTE, WHICH IS WHY THAT LIMIT IS NOW PRINTED RATHER THAN DOCUMENTED — 2026-09-05.
`level_calibration` fits its level map ON the annotated rows and writes a recalibrated table applied
corpus-wide BY DESIGN, so an arm carries the gold through the TABLE IT WAS HANDED rather than through the
rows it kept — and records `calibrated_targets: False` while doing it. A nine-member supervision ladder
was read here and printed `none of them trained on these cells`. Eight of the nine carried one of these
routes, and `recal_dense`, the row identified as the clean exception, carried this one. The claim on the
page was true about the table and false about the supervision, and every reader of that page including its
author took it for the second. `gold_fitted` names the route beside the number now; it still does not
refuse, because the reason above has not changed. See knee-vrioa.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from kneemri.evaluation.scores import LabelScores

logger = logging.getLogger(__name__)

HELD = "True"
"""What a run recorded for a flag it was given, the options table storing every flag as a string."""

CORPUS_LABELS = "report_labels_v2.csv"
"""The report-derived corpus table. Any OTHER table a run was handed is derived, and every derived table
this project has built was derived using the gold cells — so the unknown one is flagged, not cleared."""

READABLE_MIN = 5
"""Cases a finding needs on BOTH sides of 58 studies before a per-label AUC is printed at all.

`label_power` puts the per-label standard error here at 0.04-0.09 with tens of positives; below five the
number is a permutation of a handful of studies and its interval spans most of the unit line. Printing it
would put a figure on the page that no reading can use, next to figures that some can."""


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

    def gold_fitted(self, checkpoint: Path) -> str | None:
        """The route by which a run's supervision carries a constant fitted on the gold cells, or None.

        `withheld` guards the TABLE — whether the annotated ROWS sat in the training loss. Three routes
        put those same 696 cells into a run's TARGETS without ever putting the rows in the table, and all
        three pass `withheld` cleanly. Two are flags (`--calibrated-targets` invokes
        `Supervision.MEASURED_AGREEMENT`, hardcoded from these cells, and the hold drops the rows without
        refitting it); the third is the label table itself, which is why a flag check alone missed it.

        NOT A REFUSAL, deliberately — this module's header argues the case and it still holds: calibration
        moves targets toward annotation-space, which is the mechanism working rather than the score
        leaking, and `level_agreement.stability` prices four of five constants under 2 sigma. It is a limit
        on how the number may be READ, so it is reported beside the number instead of replacing it with an
        error the caller would have to route around to see anything at all.
        """
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        options = saved.get("options", {})
        if str(options.get("calibrated_targets", "")) == HELD:
            return "--calibrated-targets (Supervision.MEASURED_AGREEMENT, fitted on these cells)"
        if str(options.get("finding_calibration", "")) == HELD:
            return "--finding-calibration (per-finding rates fitted on these cells)"
        table = Path(str(options.get("labels", ""))).name
        if table and table != CORPUS_LABELS:
            return f"--labels {table} (derived, not the corpus {CORPUS_LABELS})"
        return None

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

    def joined(self, saved: list[Path], names: list[str]) -> list[tuple[Path, NDArray[np.float64]]]:
        """Readings a DIFFERENT family already wrote, aligned onto the studies this run scored.

        A foreign member states its own feed — a volumetric trunk has no read size and a windowed one has
        no volume — so it cannot be rebuilt through `MemberReadings` here without an `if` whose branches
        disagree about their arguments. It is scored by its own gate, which writes the columns, and this
        joins them BY NAME. The alignment is the point: the gate reads the annotated studies in the order
        `train.csv` lists them and this run reads them in corpus order, so an index-wise join would score
        every study against another study's radiologist and still return a plausible number.
        """
        joined = []
        for path in saved:
            saved_file = np.load(path)
            position = {str(name): index for index, name in enumerate(saved_file["identifiers"])}
            missing = [name for name in names if name not in position]
            if missing:
                message = f"{path.name} holds no reading for {len(missing)} of these studies, first {missing[0]}"
                raise ValueError(message)
            rows = np.array([position[name] for name in names], dtype=np.int64)
            joined.append((path, saved_file["values"][rows].astype(np.float64)))
        return joined

    def per_label_table(
        self,
        scores: "LabelScores",
        per_member: list[tuple[Path, NDArray[np.float64]]],
        annotations: NDArray[np.float32],
        *,
        readable_min: int = READABLE_MIN,
    ) -> list[str]:
        """One row per finding: how many of the 58 the radiologist marked present, and each member's AUC.

        THE COUNT IS PART OF THE READING, not decoration. A twelve-label mean over 58 studies hides that
        the columns behind it have between one and forty positives, so a column carried by two positives
        moves the mean as much as one carried by thirty. A finding with fewer than `readable_min` on
        either side is printed with its count and NO number at all — an AUC over two positives is defined
        arithmetically and says nothing, and printing it invites exactly the pairwise claim the paired
        intervals exist to refuse.
        """
        from core.findings import TARGETS  # noqa: PLC0415 - CLI-only
        from kneemri.evaluation.scores import MIDPOINT  # noqa: PLC0415 - CLI-only

        header = f"{'finding':20s} {'n+':>4s}  " + "  ".join(f"{member.stem[:18]:>18s}" for member, _ in per_member)
        lines = [header]
        for column, target in enumerate(TARGETS):
            present = (annotations[:, column] > MIDPOINT).astype(np.int64)
            positives, negatives = int(present.sum()), int(len(present) - present.sum())
            if min(positives, negatives) < readable_min:
                lines.append(f"{target:20s} {positives:4d}  unreadable at n={len(present)}")
                continue
            cells = "  ".join(
                f"{scores.area_under_curve(values[:, column], present):18.4f}" for _, values in per_member
            )
            lines.append(f"{target:20s} {positives:4d}  {cells}")
        return lines

    def cross(
        self,
        scores: "LabelScores",
        values: NDArray[np.float64],
        annotations: NDArray[np.float32],
        *,
        readable_min: int = READABLE_MIN,
    ) -> list[str]:
        """One member's every posterior column read against every GOLD column — the off-diagonal matrix.

        WHY THE OFF-DIAGONALS ARE THE READING. A per-finding table answers "does our synovitis column
        find the radiologist's synovitis". It cannot answer "does our EFFUSION column find it", and that
        is a different question with a different consequence: radiological convention writes synovial
        thickening into the effusion sentence, so a report-trained effusion head may have learned the
        compartment rather than the fluid. If it reads gold synovitis better than our synovitis head
        does, the serving fix is a column mix and it needs no retrain at all.

        THE DIAGONAL IS THE PER-LABEL TABLE, and printing it inside the same grid is deliberate — an
        off-diagonal cell means nothing except against the diagonal cell in its own COLUMN, which is the
        score the finding's own head achieves. A cell above its column's diagonal is the claim.

        A gold column too rare to read is withheld entirely rather than printed at n=2, on the same floor
        `per_label_table` uses and for the same reason.
        """
        from core.findings import TARGETS  # noqa: PLC0415 - CLI-only
        from kneemri.evaluation.scores import MIDPOINT  # noqa: PLC0415 - CLI-only

        present = [(annotations[:, column] > MIDPOINT).astype(np.int64) for column in range(len(TARGETS))]
        readable = [
            column
            for column in range(len(TARGETS))
            if min(int(present[column].sum()), len(present[column]) - int(present[column].sum())) >= readable_min
        ]
        lines = [f"{'posterior \\ gold':20s}" + "".join(f"{TARGETS[column][:7]:>8s}" for column in readable)]
        for row in range(len(TARGETS)):
            cells = "".join(f"{scores.area_under_curve(values[:, row], present[column]):8.3f}" for column in readable)
            lines.append(f"{TARGETS[row]:20s}{cells}")
        return lines

    def paired(
        self,
        scores: "LabelScores",
        per_member: list[tuple[Path, NDArray[np.float64]]],
        baseline: Path,
        annotations: NDArray[np.float32],
    ) -> list[str]:
        """Each member's difference from a baseline NAMED ON THE COMMAND LINE, with a paired interval.

        A bare mean over 58 studies is not a reading. `label_power` puts the per-label standard error here
        at 0.04-0.09, and six members spanning 0.014 will never separate pairwise on that denominator — so
        the means alone cannot say whether an ordering difference is a fact about the members or about the
        draw. The paired interval is what answers it: two members scored on the SAME 58 studies share the
        difficulty almost entirely, and the difference does not carry it.

        It exists because a bare six-member table read Spearman 0.14 against the proxy ordering and was
        about to be written up as the gold ruler REORDERING members. Under the intervals the whole apparent
        reorder is one pair moving +0.0035 inside a band twelve times its width, and a low rank correlation
        is what ranking six indistinguishable members produces by construction.

        THE BASELINE IS AN ARGUMENT, NOT THE WINNER OF THIS READ. `LabelScores.paired_mean_interval` is
        honest for one pair named in advance and selection-optimistic on the top row of a table it was used
        to sort. Defaulting to the FIRST checkpoint given keeps that choice in the caller's hands where it
        can be pre-registered, rather than silently picking whichever member happened to score highest.
        """
        reference = dict(per_member)[baseline]
        lines = [f"paired against {baseline.stem} (95% interval over studies, resampled):"]
        for member, values in per_member:
            if member == baseline:
                continue
            low, high = scores.paired_mean_interval(reference, values, annotations)
            delta = scores.mean(values, annotations) - scores.mean(reference, annotations)
            separates = "clears" if low > 0.0 or high < 0.0 else "INSIDE the interval — no pairwise claim"
            lines.append(f"  {member.stem:28s} {delta:+.4f} [{low:+.4f}, {high:+.4f}] {separates}")
        return lines


def main() -> None:
    """Read the given members over the annotated studies and report their agreement with the radiologist."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--checkpoints", type=Path, nargs="+", required=True, help="--hold-annotations runs")
    command.parser.add_argument("--device", default=VisibleCards().device().type, choices=("cuda", "cpu"))
    command.parser.add_argument("--overlap", action="store_true", help="read overlapping windows, as the kit does")
    command.parser.add_argument("--tta", action="store_true", help="average over rotations as the curve can")
    command.parser.add_argument("--against", type=Path, default=None, help="baseline for paired deltas; default first")
    command.parser.add_argument("--add", type=Path, nargs="*", default=[], help="npz readings from a foreign gate")
    command.parser.add_argument("--per-label", action="store_true", help="one row per finding, with its count")
    command.parser.add_argument("--cross", action="store_true", help="each posterior column against every gold column")
    arguments = command.parsed()

    from kneemri.evaluation.scores import LabelScores  # noqa: PLC0415 - CLI-only
    from kneemri.inference.vote import RankAverage  # noqa: PLC0415 - CLI-only

    root = command.rooted(arguments)
    identifiers, truth = GoldRows().annotated(root.raw() / "train.csv")
    views = SpatialViews() if arguments.tta else None
    reader = AnnotatedReadings(root, torch.device(arguments.device), views)
    names, per_member = reader.readings(arguments.checkpoints, identifiers, overlap=arguments.overlap)
    per_member = per_member + reader.joined(arguments.add, names)

    position = {name: index for index, name in enumerate(identifiers)}
    annotations = truth[np.array([position[name] for name in names], dtype=np.int64)]
    scores = LabelScores()
    combined = RankAverage().combine([values for _, values in per_member])
    routes = [(path, reader.gold_fitted(path)) for path in arguments.checkpoints]
    flagged = [f"  {path.stem:28s} {route}" for path, route in routes if route is not None]
    caveat = [
        "",
        f"{len(flagged)} of them carry a constant FITTED on these cells, so their score here is agreement",
        "between two rulers that share an origin, never a confirmation (knee-vrioa):",
        *flagged,
    ]
    origin = caveat if flagged else []
    breakdown = ["", *reader.per_label_table(scores, per_member, annotations)] if arguments.per_label else []
    matrices: list[str] = []
    for member, values in per_member if arguments.cross else []:
        matrices += ["", f"cross-column AUC, {member.stem} (rows: our posterior; columns: gold truth)"]
        matrices += reader.cross(scores, values, annotations)
    command.report(
        [f"{len(names)} annotated studies, {len(per_member)} member(s), none of them TRAINED on these cells"]
        + [f"{member.stem:28s} {scores.mean(values, annotations):.4f}" for member, values in per_member]
        + [f"{'combined':28s} {scores.mean(combined, annotations):.4f}"]
        + ([""] + reader.paired(scores, per_member, arguments.against or arguments.checkpoints[0], annotations))
        + breakdown
        + matrices
        + origin
    )


if __name__ == "__main__":
    main()
