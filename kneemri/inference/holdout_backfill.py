"""Re-derive an already-finished run's holdout predictions: `python -m kneemri.inference.holdout_backfill`.

Every run fitted since knee-mx17 keeps the array its per-label numbers were read off, so two such arms
can be compared against their OWN paired floor. The 65 runs already on disk predate it: they kept eight
witness studies and dropped the rest, so `run_comparison` still marks them against the unpaired bound
and the ledger's per-label findings stay UNREAD — neither kept nor killed. That is knee-3vh's second
acceptance clause, and it needs no training slot, only an inference pass.

TWO THINGS DECIDE WHETHER THE RESULT MEANS ANYTHING.

The split must be the run's OWN. It is taken from the recipe the run recorded — folds, split seed and
fold index — and never from this command's flags, because a holdout assembled from the wrong fold is
not an error anywhere downstream: the join by study name simply shrinks to whatever happens to overlap
and reports a tighter interval over fewer studies. A run that recorded no recipe is REFUSED here rather
than guessed at.

And a re-derivation is not the training array. The same weights read the same pixels, but through the
submission path rather than the training one, and on whatever device is free — which is accurate to the
cross-precision tolerance, roughly 4x eps, and that is WIDER than the +/-0.016 half-widths a paired
floor quotes. So the record is written as `DERIVED`, and `run_comparison` prints any floor computed
from one as INDICATIVE. The record says what it is; nothing downstream has to remember which runs were
back-filled.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from core.paths import DataRoot
from core.wakelock import Wakelock
from kneemri.corpus.folds import ReportFolds
from kneemri.corpus.study_cache import CachedCorpus
from kneemri.device import VisibleCards
from kneemri.inference.holdout_predictions import HoldoutPredictions
from kneemri.inference.holdout_slab import HoldoutSlab
from kneemri.inference.members import Member
from kneemri.inference.prediction_cache import PredictionCache
from kneemri.labels.targets import Targets
from kneemri.runs.holdout_record import DERIVED, UNKNOWN_EPOCH, HoldoutRecord
from kneemri.runs.recorded_run import RecordedRun
from kneemri.runs.run_metrics import RunMetrics

logger = logging.getLogger(__name__)

NO_HOLDOUT_FOLD = -1
"""What a full-data fit records: it held nothing out, so there is nothing to re-derive."""


class UnbackfillableError(ValueError):
    """Raised when a run cannot have its holdout re-derived, with the reason it cannot.

    A refusal per run rather than one for the walk: most of the ledger predates the recipe, and a
    command that stops at the first such run would back-fill nothing. Each reason is printed, because
    "45 runs skipped" is the kind of silence this project keeps paying for.
    """


@dataclass(frozen=True)
class Backfill:
    """One run that CAN be re-derived: its checkpoint, the recipe it recorded, and the epoch kept."""

    checkpoint: Path
    recipe: dict[str, str]
    epoch: int

    @classmethod
    def of(cls, checkpoint: Path) -> "Backfill":
        """The run's own account of itself, or a refusal naming what cannot be reconstructed from it.

        WHICH RECORD ANSWERS WHAT is `RecordedRun`'s judgement, not this one's — the same question is
        asked by `run_comparison`, and the two used to answer it differently. What belongs here is only
        what a RE-DERIVATION additionally needs: a split it can rebuild, and a run that held something
        out to rebuild it for.

        A run that recorded no epoch reads `UNKNOWN_EPOCH` and is still back-filled — the epoch names
        WHICH weights predicted, and the weights are on disk regardless. Refusing here would drop 36
        re-derivable runs to protect a field nothing joins on yet, and guessing it is what stamped
        `wsfold0_e6` as epoch 11.
        """
        recorded = RecordedRun.of(RunMetrics(checkpoint))
        recipe = recorded.recipe
        if recipe is None:
            message = (
                f"{checkpoint.stem}: no recipe recorded in its metrics or its checkpoint, "
                "so its split cannot be reconstructed"
            )
            raise UnbackfillableError(message)
        # BEFORE the epoch check, because a full-data fit fails both and only this reason tells the
        # reader anything: it records NaN AUCs precisely BECAUSE it held nothing out.
        if int(recipe.get("fold", NO_HOLDOUT_FOLD)) == NO_HOLDOUT_FOLD:
            message = f"{checkpoint.stem}: a full-data fit held nothing out"
            raise UnbackfillableError(message)
        epoch = recorded.kept_epoch
        return cls(checkpoint=checkpoint, recipe=recipe, epoch=UNKNOWN_EPOCH if epoch is None else epoch)

    def split(self, identifiers: list[str], train_table: Path) -> NDArray[np.int64]:
        """The studies THIS run held out, from its own recorded folds, split seed and fold index."""
        folds = ReportFolds(int(self.recipe["folds"]), int(self.recipe["split_seed"]))
        _, holdout = folds.holdout(folds.assign(str(train_table), identifiers), int(self.recipe["fold"]))
        return holdout

    def ruler(self, identifiers: list[str], train_table: Path) -> Targets:
        """The table this run was SCORED against, which is `--eval-labels` when it named one.

        The same distinction the training path makes: an arm whose variable was the training labels was
        deliberately scored against a table it did not train on, and pairing it against a control scored
        by the other table is arithmetic on two different questions — which `HoldoutReading.aligned`
        refuses outright once the truth travels with the predictions.
        """
        evaluation = self.recipe.get("eval_labels")
        scoring = Path(evaluation) if evaluation and evaluation != "None" else Path(self.recipe["labels"])
        return Targets.load(scoring, train_table, identifiers)


class HoldoutBackfill:
    """The inference pass that gives an already-run arm the per-study record it never kept."""

    def __init__(self, root: DataRoot, device: torch.device, *, overlap: bool = False) -> None:
        self.root = root
        self.predictions = HoldoutPredictions(device)
        self.overlap = overlap

    def derive(self, member: Member, backfill: Backfill, corpus: CachedCorpus, identifiers: list[str]) -> int:
        """Re-derive one run's holdout predictions and write them beside its checkpoint.

        The member's OWN read size, precision and window — through `Member`, which is the type the
        submission path uses, so this cannot grow its own reading of a checkpoint and drift from the one
        that ships. Returns how many studies were written.

        The corpus is passed in rather than loaded here for the same reason `HoldoutPredictions.cached`
        takes a slab: everything expensive sits behind it, so a run whose predictions are already cached
        costs no decode at all — and this half can then be tested without 35 GB of pixels.
        """
        train_table = self.root.raw() / "train.csv"
        holdout = backfill.split(identifiers, train_table)
        truth = backfill.ruler(identifiers, train_table).scores[holdout]
        studies = HoldoutRecord.named(identifiers, holdout)

        window = member.window(corpus.slices, overlap=self.overlap)
        # ANNOUNCED BEFORE THE PASS RATHER THAN AFTER IT. A walk over 55 members costs hours on a CPU
        # and printed nothing until one finished, so the only reading available part-way through was
        # "no output yet" — which is what a hung process prints too.
        logger.info(
            "%s: deriving %d studies at %d px over %d window(s)",
            backfill.checkpoint.stem,
            len(studies),
            member.read_size,
            len(window.starts),
        )
        store = PredictionCache(self.root.processed() / "holdout_predictions", studies, self.predictions.views)
        predicted = self.predictions.cached(member, HoldoutSlab(corpus, holdout), store, window)

        HoldoutRecord(backfill.checkpoint).write(studies, predicted, truth, backfill.epoch, source=DERIVED)
        return len(studies)


def main() -> None:
    """Back-fill the arms already run: `python -m kneemri.inference.holdout_backfill --checkpoints *.pt`."""
    parser = argparse.ArgumentParser(description=HoldoutBackfill.__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    parser.add_argument("--device", default=VisibleCards().device().type)
    parser.add_argument("--overlap", action="store_true", help="read overlapping windows, as the kit does")
    parser.add_argument("--refresh", action="store_true", help="re-derive runs that already kept a record")
    parser.add_argument(
        "--restamp", action="store_true", help="correct an existing record's epoch only, with no inference"
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    backfill = HoldoutBackfill(
        DataRoot.from_config(arguments.paths), torch.device(arguments.device), overlap=arguments.overlap
    )
    # The walk, not each member: a suspend between two members costs the same wall-clock as one inside
    # a member, and the walk over 55 of them is what runs for hours unattended (knee-wyzj).
    with Wakelock():
        for checkpoint in arguments.checkpoints:
            # A restamp is only ever asked of a record that EXISTS, so it passes this gate for the same
            # reason a refresh does — the gate keeps a plain walk from re-deriving what is done.
            if HoldoutRecord(checkpoint).exists() and not (arguments.refresh or arguments.restamp):
                logger.info("%s: already keeps its predictions", checkpoint.stem)
                continue
            try:
                plan = Backfill.of(checkpoint)
            except UnbackfillableError as refusal:
                logger.info("skipped %s", refusal)
                continue
            if arguments.restamp:
                moved = HoldoutRecord(checkpoint).restamp(plan.epoch)
                logger.info("%s: epoch %s", checkpoint.stem, f"restamped to {plan.epoch}" if moved else "already right")
                continue
            member = Member.from_checkpoint(checkpoint)
            corpus, identifiers = member.cache(backfill.root.processed()).load()
            written = backfill.derive(member, plan, corpus, identifiers)
            logger.info("%s: %d studies re-derived at epoch %d", checkpoint.stem, written, plan.epoch)


if __name__ == "__main__":
    main()
