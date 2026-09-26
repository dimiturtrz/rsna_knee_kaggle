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
submission path rather than the training one, and on whatever device is free, so the two disagree at
the cross-precision tolerance. MEASURED 2026-09-17 on `labelwise_depth24_drop40_long_s2026`, 909
studies, cuda, by re-deriving a run that already kept a TRAINED record and diffing the two arrays:

    per-cell |delta|   max 0.016822   mean 0.000964   p99 0.006409     (fp16 eps = 9.77e-04)
    macro AUC          trained 0.867917   derived 0.867955   delta +0.000038
    worst label        Lateral OA +0.0004, against the 0.0062 paired floor

The two numbers live in DIFFERENT UNITS and must not be compared: a per-cell prediction delta is not a
score half-width. The per-cell max is 17x eps, not the 4x this docstring asserted unmeasured; the SCORE
it produces is 0.00004, a hundred and fifty times inside the floor. So a `DERIVED` record is usable for
a paired floor on the evidence, not merely by tolerance — and the reason it is still stamped `DERIVED`,
and still printed by `run_comparison` as INDICATIVE, is that this is one member on one device. A member
whose trunk is deeper diverges further (`witness_check` reads 1.6-2.4x eps on four members and 5.8x on
the project's largest), so the stamp records the provenance rather than a tolerance nobody re-measures.
"""

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from core import cli
from core.data.corpus.folds import ReportFolds
from core.data.corpus.slot_reader import GROUP, SLICE_BAND
from core.data.corpus.study_cache import CachedCorpus
from core.data.labels.targets import Targets
from core.paths import DataRoot
from core.wakelock import Wakelock
from kneemri.device import VisibleCards
from kneemri.inference.ensemble import SharedRead
from kneemri.inference.holdout_predictions import HoldoutPredictions
from kneemri.inference.holdout_slab import HoldoutSlab
from kneemri.inference.members import Member
from kneemri.inference.prediction_cache import PredictionCache
from kneemri.models.backbone import ENCODERS
from kneemri.models.fingerprint import FINGERPRINT
from kneemri.runs.holdout_record import DERIVED, UNKNOWN_EPOCH, HoldoutRecord
from kneemri.runs.recorded_run import RecordedRun
from kneemri.runs.run_record import RunRecord

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
        recorded = RecordedRun.of(RunRecord(checkpoint))
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


@dataclass(frozen=True)
class ReadSize:
    """Serving an already-fitted member at a size it was not fitted at, priced on its own holdout.

    The efficiency question is what a smaller read COSTS in accuracy, and the only honest way to ask it
    of a member that exists is to re-derive its holdout through the submission path at that size. The
    override is `train_size`, never `img`: `img` names the cache the member shares, so moving it feeds
    resampled pixels to an encoder still running at the old resolution and prices nothing (knee-d2fc).

    The record is written beside the checkpoint under a SIZED name, so a price never overwrites the
    member's native record — which is the number every ensemble and every seat search reads.
    """

    pixels: int

    def read(self, fitted: Member) -> Member:
        """The same weights, read at this size."""
        return fitted.resized(self.pixels)

    def destination(self, checkpoint: Path) -> Path:
        """Where the priced record goes: beside the checkpoint, named for the size that produced it."""
        return checkpoint.with_name(f"{checkpoint.stem}_r{self.pixels}{checkpoint.suffix}")


@dataclass(frozen=True)
class Reading:
    """Which reading a walk prices: the member's native one, or a size and a depth it was not fitted at."""

    sizing: ReadSize | None = None
    regroup: SharedRead | None = None

    def native(self) -> bool:
        """Is this the reading the member was fitted at — the only one whose price may be stamped?"""
        return not (self.sizing or self.regroup)

    def destination(self, checkpoint: Path) -> Path:
        """The checkpoint whose record this reading writes, sized and re-grouped as asked."""
        sized = self.sizing.destination(checkpoint) if self.sizing else checkpoint
        return self.regroup.beside(sized) if self.regroup else sized

    def load(self, checkpoint: Path, processed: Path) -> tuple[Member, CachedCorpus, list[str]]:
        """The member at this reading and the decoded corpus it reads from.

        The re-group runs BEFORE the cache is resolved and the resize after it, because the two
        overrides act on different halves of the read: slices and band decide WHICH files are decoded,
        so they name a different cache; a size is applied to pixels already decoded, so it must not
        move the cache the member shares (knee-d2fc).
        """
        member = Member.from_checkpoint(checkpoint)
        member = self.regroup.regrouped(member) if self.regroup else member
        corpus, identifiers = member.cache(processed).load()
        return (self.sizing.read(member) if self.sizing else member), corpus, identifiers


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
        train_table = self.root.train_table()
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

    def stamp(self, member: Member, checkpoint: Path) -> bool:
        """Give a checkpoint that has no weight fingerprint one, computed from the weights on disk.

        A WEIGHT SOUP ARRIVES WITHOUT ONE BY DESIGN, AND COULD THEREFORE BE SCORED BUT NEVER SHIPPED.
        `weight_soup` drops the fingerprint rather than copying it forward, because a fingerprint
        witnesses a specific set of tensors and an average is none of them; its docstring records that a
        soup "is re-fingerprinted" when it earns a seat, and that step did not exist. `package_members`
        is a file copy with no device, and indexes `checkpoint["fingerprint"]` directly, so every one of
        the ten soups on disk raised `KeyError` at packaging with every other gate green (knee-emyky).
        Here because this is the one pass that already rebuilds the member on a card.

        The soup's OTHER missing field, its `holdout`, is not written here: pricing a record is
        `kneemri.evaluation`'s arithmetic and inference may not import the layer that scores it.
        `ledger75 --price` writes that one, off the same row it prints.

        Returns whether anything was written. A checkpoint that already carries a fingerprint is left
        alone — a fingerprint is a witness of the weights that were measured, and recomputing one
        silently is how a member gets a fresh witness for weights nobody re-read.
        """
        saved = dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        if len(np.asarray(saved.get("fingerprint", []), np.float32)) > 0:
            return False

        encoder = ENCODERS[member.variant]
        model = self.predictions.load(checkpoint, encoder.load(), encoder.dim, member.slot_count)
        saved["fingerprint"] = model.fingerprint(member.read_size, GROUP, self.predictions.device).tolist()
        saved["fingerprint_config"] = {
            "img": member.read_size,
            "n_slot": model.n_slot,
            "group": GROUP,
            "seed": FINGERPRINT.seed,
        }
        # Never in place: a checkpoint half-written by a crash is a member that loads as far as the
        # truncation and then raises, and this one is 89 MB over a copy.
        staged = checkpoint.with_suffix(".stamping")
        torch.save(saved, staged)
        staged.replace(checkpoint)
        logger.info("%s: stamped weight fingerprint", checkpoint.stem)
        return True

    def visit(self, checkpoint: Path, reading: Reading, *, refresh: bool = False, restamp: bool = False) -> bool:
        """One checkpoint of the walk, returning whether its holdout was re-derived."""
        destination = reading.destination(checkpoint)
        # Before the holdout gate, not inside it. `stamp` is a no-op for a checkpoint already witnessed,
        # so the walk costs nothing — and a soup of FULL-DATA fits holds nothing out, so it refuses
        # backfill below and reaches no other stamping path in the repo (knee-emyky).
        if reading.native():
            self.stamp(Member.from_checkpoint(checkpoint), destination)
        # A restamp is only ever asked of a record that EXISTS, so it passes this gate for the same
        # reason a refresh does — the gate keeps a plain walk from re-deriving what is done.
        if HoldoutRecord(destination).exists() and not (refresh or restamp):
            logger.info("%s: already keeps its predictions", checkpoint.stem)
            return False
        try:
            plan = Backfill.of(checkpoint)
        except UnbackfillableError as refusal:
            logger.info("skipped %s", refusal)
            return False
        if restamp:
            moved = HoldoutRecord(checkpoint).restamp(plan.epoch)
            logger.info("%s: epoch %s", checkpoint.stem, f"restamped to {plan.epoch}" if moved else "already right")
            return False
        return self._derived(checkpoint, replace(plan, checkpoint=destination), reading)

    def _derived(self, checkpoint: Path, plan: Backfill, reading: Reading) -> bool:
        member, corpus, identifiers = reading.load(checkpoint, self.root.processed())
        written = self.derive(member, plan, corpus, identifiers)
        logger.info("%s: %d studies re-derived at epoch %d", checkpoint.stem, written, plan.epoch)
        # NATIVE PASSES ONLY. A sized or re-grouped record prices the member at a reading it was not
        # fitted at, and stamping the checkpoint's `holdout` from one would replace the number every
        # ensemble and seat search reads with a price of something else.
        if reading.native():
            self.stamp(member, plan.checkpoint)
        return True


def main() -> None:
    """Back-fill the arms already run: `python -m kneemri.inference.holdout_backfill --checkpoints *.pt`."""
    command = cli.Command(HoldoutBackfill.__doc__, logger)
    command.parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    command.parser.add_argument("--device", default=VisibleCards().device().type)
    command.parser.add_argument("--overlap", action="store_true", help="read overlapping windows, as the kit does")
    command.parser.add_argument("--refresh", action="store_true", help="re-derive runs that already kept a record")
    command.parser.add_argument(
        "--restamp", action="store_true", help="correct an existing record's epoch only, with no inference"
    )
    command.parser.add_argument(
        "--read-size", type=int, default=0, help="price the member at this many pixels, into a sized record"
    )
    command.parser.add_argument(
        "--served-slices", type=int, default=0, help="price the member at this depth, into a re-grouped record"
    )
    command.parser.add_argument(
        "--served-band",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="price the member at this band, into a re-grouped record; needs --served-slices",
    )
    arguments = command.parsed()

    band = (arguments.served_band[0], arguments.served_band[1]) if arguments.served_band else SLICE_BAND
    reading = Reading(
        ReadSize(arguments.read_size) if arguments.read_size else None,
        SharedRead(arguments.served_slices, band) if arguments.served_slices else None,
    )
    backfill = HoldoutBackfill(command.rooted(arguments), torch.device(arguments.device), overlap=arguments.overlap)
    # The walk, not each member: a suspend between two members costs the same wall-clock as one inside
    # a member, and the walk over 55 of them is what runs for hours unattended (knee-wyzj).
    with Wakelock():
        for checkpoint in arguments.checkpoints:
            backfill.visit(checkpoint, reading, refresh=arguments.refresh, restamp=arguments.restamp)


if __name__ == "__main__":
    main()
