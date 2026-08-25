"""What a trained member records about itself, so a checkpoint can say what produced it."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from core.data.corpus.crop_centre import Crop
from core.data.corpus.slot_reader import GROUP, Band
from core.data.corpus.slots import Routing
from core.findings import TARGETS
from kneemri.models.backbone import Encoder
from kneemri.models.fingerprint import FINGERPRINT
from kneemri.models.model import StudyModel

RECORDED_AS = {
    "image_size": "img",
    "band_low": "band",
    "band_high": "band",
    "scheme": "slots",
}
"""Run settings the config carries under a DIFFERENT name, so a name walk cannot see them.

The walk in the unit gate moves one option and demands the config field of THAT NAME follow, which is
the check that catches a value stamped from the wrong option. It can only see the fields whose names
agree, and these four are the ones that do not: a band travels as the pair it was resolved into, a
scheme as the slot list it names, and `image_size` under the `img` the serve path has always read.
"""

NOT_IN_CONFIG = {
    "paths": "the data root, machine-specific and gitignored",
    "labels": "the label table, an input rather than a property of the weights",
    "eval_labels": "the holdout table, same",
    "out": "where the run wrote, which the file's own location already answers",
    "epochs": "run length — `epochs_done` records what was actually reached, beside the kept epoch",
    "patience": "when the run stopped, which the kept epoch already states",
    "studies": "corpus truncation for a smoke; `cache_studies` records WHICH cache was read",
    "train_fraction": "how much of the TRAINING half was fitted on — a property of the run's data, not "
    "of the function a rebuild reconstructs, exactly as `fold` and `studies` are. The recipe block "
    "carries it, which is where a scaling point's provenance is read from, and the run's own split line "
    "prints the study count it produced",
    "batch": "a resource knob: how many studies a step held changes no weight's meaning",
    "accumulate": "the same knob one level up — micro-batches per step trade memory for time and leave "
    "the rebuilt function untouched; the recipe block carries it, which is where the landed-check reads",
    "vram_gb": "a memory cap read against the card the run had",
    "seed": "seeds and folds are banned as arms here, and the recipe block still carries it",
    "split_seed": "same, and it moves the split rather than the member",
    "folds": "banned for the same reason",
    "fold": "which half was held out — a property of the SCORE, and `holdout` is what serves it",
    "one_cycle": "a schedule: it changes how the weights were reached, never how they are rebuilt",
    "weight_floor": "supervision policy, same",
    "calibrated_targets": "supervision policy, same",
    "severity_weight": "supervision policy, same — it grades the loss weight and leaves the head alone",
    "hold_annotations": "supervision policy, same — it decides which cells the loss READ and touches no "
    "parameter, so the head a rebuild serves is byte-identical",
    "grade_ladder": "objective policy — the thresholds are subtracted INSIDE the loss, so the head still "
    "emits the one per-label logit a rebuild serves",
    "side_weight": "objective policy — the term compares two of the twelve logits against each "
    "other INSIDE the loss and adds no parameter, so the head a rebuild serves is byte-identical",
    "focus_label": "objective policy — it zeroes eleven columns of the loss WEIGHT and touches no "
    "parameter, so the twelve-logit head a rebuild serves is byte-identical. The hazard it carries is "
    "not a rebuild hazard: a focused member's other eleven columns are untrained, so it is an "
    "INSTRUMENT and never a roster seat, which its arm states and `members.roster` enforces by simply "
    "not naming it",
    "per_label_loss": "objective policy, same",
    "normalise_by_count": "objective policy, same",
    "intensity_aug": "an input augmentation applied while fitting, never while serving",
    "recompute_activations": "a memory-for-compute trade inside the backward pass",
    "compile_encoder": "a compilation of the same function — `load_trained` strips its prefix",
    "report_vectors": "the report head is ALWAYS built, so no rebuild can ask whether it was trained",
    "distill_weight": "objective policy — the term pulls the SAME twelve logits toward a teacher's "
    "ordering inside the loss and adds no parameter, so the head a rebuild serves is byte-identical",
    "auc_margin": "objective policy — it swaps WHICH scalar the same twelve logits descend, adds no "
    "parameter (its per-label constants are solved in closed form inside the loss, never stored), and "
    "leaves the head a rebuild serves byte-identical",
    "dissent_weight": "objective policy — it re-weights the mass of the SAME label-ordering term by how "
    "badly a teacher pool orders each pair, inside the loss, adding no parameter; the head a rebuild "
    "serves is byte-identical",
    "teacher_targets": "the ensemble that taught it, which is a fact about how these weights were "
    "REACHED — a rebuild loads the student, and the teacher is not part of the function it is",
    "encoder_weights": "where the encoder STARTED, which the fitted weights have already absorbed — a "
    "rebuild loads this member, not the initialisation it grew out of",
    "encoder_adapt_depth": "how much of that starting point was taken — a fact about the run's history, "
    "not about the function these weights are, and a rebuild loads the fitted encoder whole",
}
"""Every run setting the config does NOT carry, each with WHY — a decision record.

THE CONFIG IS THE REBUILD CONTRACT: `StudyDesign.from_config` and `Member`'s accessors read it to
reconstruct the function these weights ARE, and a value that lands here while a consumer reads a module
default is the shape that has cost this project nine defects. So the question a new knob has to answer
is not "is it interesting" but "must a rebuild know it" — and everything above answers no, either
because it describes how the weights were REACHED or because it describes the machine that reached
them. None of it is lost: `payload["options"]` stamps the whole recipe, and `test_recipe` asserts that
block holds every field.

Made TOTAL and asserted at the unit gate, next to `Landed.NOT_A_SWITCH` and
`switch_coverage.NOT_A_RECIPE`. A knob nothing records and nothing writes off is indistinguishable
from one nobody has thought about yet, and the second is how a member gets rebuilt as a different
function with every fingerprint still passing against its own reading.

SPELLED HERE rather than imported from `TrainingOptions`, for the reason `RunOptions` is structural:
train.py imports this module, so naming its dataclass would close the cycle.
"""


class RunOptions(Protocol):
    """The run settings a checkpoint has to record, as a shape rather than a named class.

    Structural, so this module does not import the CLI that produces it — train.py imports this one,
    and a concrete type here would make that a cycle. It also lets a test construct the fields without
    a parsed command line.
    """

    @property
    def recipe(self) -> dict[str, str]:
        """Every flag the run was given, as strings — stamped whole, never re-derived here."""
        ...

    @property
    def image_size(self) -> int: ...

    @property
    def crop(self) -> Crop:
        """The physical crop the RUN read, which the checkpoint must record rather than assume."""
        ...

    @property
    def band(self) -> Band:
        """The slice band the RUN sampled over — recorded, and read by the serve path to build a reader."""
        ...

    @property
    def routing(self) -> Routing:
        """The router the RUN used, for the same reason: a member served under another sees other series."""
        ...

    @property
    def encoder_layer(self) -> int: ...

    @property
    def train_size(self) -> int: ...

    @property
    def cache_studies(self) -> int:
        """How many studies the cache the RUN read held — which cache, not just its shape."""
        ...

    @property
    def slices(self) -> int: ...

    @property
    def pool(self) -> str:
        """How a slice's patch tokens become one vector, which fixes the head's width when it is rebuilt."""
        ...

    @property
    def slice_pooling(self) -> str: ...

    @property
    def slice_regions(self) -> int: ...

    @property
    def through_plane(self) -> bool: ...

    @property
    def slot_pooling(self) -> str: ...

    @property
    def anatomical_prior(self) -> bool: ...

    @property
    def ema(self) -> bool: ...

    @property
    def bf16(self) -> bool: ...

    @property
    def lora_rank(self) -> int: ...

    @property
    def rank_weight(self) -> float: ...

    @property
    def pair_memory(self) -> int: ...

    @property
    def report_weight(self) -> float: ...

    @property
    def whole_stack(self) -> bool: ...

    @property
    def unfreeze_last(self) -> int: ...

    @property
    def head_lr(self) -> float: ...

    @property
    def backbone_lr(self) -> float: ...

    @property
    def layer_decay(self) -> float: ...


WITNESS_STUDIES = 8
"""Enough to catch a reading difference, small enough to cost nothing — a mismatch moves every row."""


@dataclass(frozen=True)
class MemberRecord:
    """Assembles a member's checkpoint: its weights, its score, and the reading it was fitted under.

    ONE assembler for both the periodic save and the final one. They used to differ — the final save
    wrote the config, the recipe and the fingerprint, while `save_partial` wrote only weights, holdout
    and epoch. That made the crash-safety net preserve a run's WORK but not its IDENTITY: the weights
    survive a kill and `MemberPackage` then refuses them, because a member with no fingerprint and no
    reading cannot be run at all. A 0.8413 member was left in exactly that state today, and the point
    of killing it early was to keep it.

    The fingerprint costs a two-study forward pass per improved epoch, which is nothing beside a
    70-minute epoch, and it is what makes a surviving checkpoint submittable rather than merely
    inspectable.

    A dataclass because these five are one thing, and because five constructor arguments written by
    hand would trip the argument-count gate for no benefit.
    """

    options: RunOptions
    encoder: Encoder
    model: StudyModel
    device: torch.device
    slot_names: list[str]

    @property
    def read_size(self) -> int:
        """The resolution the model actually saw — the cache's, unless the run resampled off it.

        The fingerprint is taken at THIS size, not at the cache's. A member trained at 392 px from a
        448 px cache and rebuilt at 448 is a different function on different pixels; recording the
        cache size alone would let it advertise a reading it was never fitted under. That failure has
        happened once already, with `slice_pooling`, and cost 2.25 against a 2e-3 tolerance.
        """
        return self.options.train_size or self.options.image_size

    @property
    def recipe(self) -> dict[str, str]:
        """The run's own account of its flags — the checkpoint's `options` block, and the metrics file's.

        THE OPTIONS OBJECT OWNS IT, and this delegates rather than re-deriving, for the reason the
        recipe is one home at all: the metrics file records it so a per-label comparison can say which
        axes moved between two runs, `train` logs it before epoch 0 so a running arm is auditable
        without waiting for its checkpoint, and a recipe that disagreed with the checkpoint's would be
        worse than none. Reaching through `record.options` at each call site instead would be three
        strangers deep and would say the same thing three times.
        """
        return self.options.recipe

    def witness(
        self,
        identifiers: list[str],
        holdout: NDArray[np.int64],
        predictions: NDArray[np.float64],
        batch: int,
    ) -> dict[str, object]:
        """What the TRAINING path predicted for a few named holdout studies.

        Named studies rather than positions, because a position means nothing without the same corpus
        index — and the whole point is to be checkable from a different process, against a cache built
        later, by code that never saw this run.

        THE BATCH IS PART OF THE CLAIM. These eight are the first eight rows of a `batch`-row forward,
        and a half-precision GEMM tiles by shape: a checker that feeds eight rows reduces in a different
        order and carries a rounding of its own. The recipe records `batch` too, but the recipe does not
        travel into a packaged member — only this dict does — so the checker could not ask for the one
        number that would let it reproduce the shape.
        """
        chosen = holdout[:WITNESS_STUDIES]
        return {
            "studies": [identifiers[position] for position in chosen],
            "predictions": predictions[: len(chosen)].tolist(),
            "batch": batch,
            "reading": "trainer.predict — the middle window of the stack, at the run's own read size",
        }

    def payload(
        self,
        state: dict[str, torch.Tensor],
        holdout: float,
        epoch: int,
        witness: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """The full checkpoint for a set of weights — identical in shape whenever it is written.

        `witness` is what the TRAINING path predicted for a handful of named holdout studies. The
        fingerprint proves the model is the model; it cannot prove that inference READS the model the
        way training did, because it fixes only image size and group. Four separate defects this
        session were exactly that — a value recorded in this config and a consumer using a default
        instead: whole-stack reading windowed to three slices, the decode resolution taken from
        SlotReader's default, the slot list taken from a module constant twice. Every one of them
        loaded cleanly, verified cleanly, and predicted from something the weights had never seen.

        A witness closes that hole because it is an end-to-end claim: re-derive these studies through
        the INFERENCE path and you must get these numbers back. Any consumer-side default that changes
        what the model reads moves them.
        """
        options = self.options
        return {
            "model": state,
            "holdout": holdout,
            "annot": float("nan"),
            "epoch": epoch,
            "epochs_done": epoch + 1,
            # The recipe, not just the image geometry. Without it a checkpoint cannot say which arm
            # produced it: two seed runs sit on disk whose one-cycle/floor/label flags are now
            # unrecoverable, so they cannot serve as a noise floor for anything.
            # str(), not the dataclass as-is: every loader here reads with weights_only=True, which
            # refuses to unpickle a Path, so saving the options raw would make the checkpoint
            # unloadable by our own code.
            "options": self.recipe,
            "targets": list(TARGETS),
            "slots": list(self.slot_names),
            "config": {
                # Inside the config so it reaches the manifest for free: package_members copies
                # config wholesale, and a witness that does not travel with the member cannot check it.
                "witness": witness or {},
                "img": options.image_size,
                "train_size": self.read_size,
                "slices": options.slices,
                # WHICH cache, not just its shape. A smoke run reads a truncated cache, whose arrays
                # differ from the full ones only in their first axis and whose index is a different
                # list of studies — so a member fitted on one and re-read against the other is scored
                # on the wrong pixels per position, with the shape and every fingerprint agreeing.
                "cache_studies": options.cache_studies,
                "group": GROUP,
                # THE RUN'S OWN CROP, not the module constant. Stamping the constant here would have
                # recorded 130 mm about the image centre for an arm trained at 80 mm about the tissue,
                # and the serve path reads this field — so the member would have been fed pixels it was
                # never fitted on, with every fingerprint still passing against its own reading.
                "crop_mm": options.crop.mm,
                "centre": options.crop.centre,
                # The RUN's router. Without this a graft-matching arm would record nothing and be
                # served under our first-maximal rule — fitted on one set of series, fed another.
                "tie": options.routing.tie,
                # THE RUN'S OWN BAND, for the reason `crop_mm` gives above and on the axis it does
                # not cover: this field was the last one stamped from a module constant, and the serve
                # path reads it to build the reader. A member fitted on a wide band and served on a
                # narrow one sees a stack it was never fitted on, with every fingerprint still passing.
                "band": options.band.recorded(),
                "slots": list(self.slot_names),
                "variant": self.encoder.variant,
                # THE RUN'S OWN POOLING, for the reason `crop_mm` gives above: `Member.pool` reads this
                # field back and it is what the serve path rebuilds `StudyDesign` from. A literal here
                # meant a focal-pooled member would be reconstructed two parts wide against three-part
                # weights — and, before this, no arm could select the focal pooling at all.
                "pool": options.pool,
                "slice_pooling": options.slice_pooling,
                # Read back by `StudyDesign.from_config`, which is the only thing that knows a maximum
                # is taken per region: the weights are identical either way, so a member fitted over
                # four regions and rebuilt over one loads perfectly and predicts from a different
                # function of the same pixels. Exactly the shape `slice_pooling` itself once had.
                "slice_regions": options.slice_regions,
                # Read back by `StudyDesign.from_config` and consumed in `slice_features`: a member
                # fitted on (i-1, i, i+1) and rebuilt on (i, i, i) loads perfectly and predicts from
                # different pixels. The fingerprint is what catches it, and only if this is stamped.
                "through_plane": options.through_plane,
                # Recorded AND read back by Member.slot_pooling and HoldoutPredictions.load. A head
                # fitted with uniform slot pooling and rebuilt with attention is a different function
                # whose weights load without complaint - the ninth instance of that shape is why the
                # ablation's field was wired to its consumers before the arm was ever queued.
                "slot_pooling": options.slot_pooling,
                "anatomical_prior": options.anatomical_prior,
                # Recorded AND read back by Member.layer. A value written here and defaulted by a
                # consumer is the shape that has cost this project nine defects.
                "encoder_layer": options.encoder_layer,
                "ema": options.ema,
                # Precision is part of what produced these weights: bf16 and fp16 are different
                # arithmetic, and a member scored under one is not strictly comparable to a control
                # measured under the other. Recorded so the ledger can say which, rather than assume.
                "bf16": options.bf16,
                # WITHOUT THIS THE CHECKPOINT IS UNLOADABLE. A LoRA member's state dict carries adapter
                # keys that exist only if `low_rank` ran first, so a loader that does not know the rank
                # builds the wrong architecture — the seventh instance of a value travelling in a config
                # while a consumer reads a default, and the first one caught before it cost anything.
                "lora_rank": options.lora_rank,
                "rank_weight": options.rank_weight,
                "pair_memory": options.pair_memory,
                "report_weight": options.report_weight,
                "whole_stack": options.whole_stack,
                # The value this run USED, never the module constant. Recording the constant made the
                # field a lie the moment the depth became a flag, and the loader reads it back to
                # decide how a member was fitted.
                "unfreeze_last": options.unfreeze_last,
                "head_lr": options.head_lr,
                "backbone_lr": options.backbone_lr,
                "layer_decay": options.layer_decay,
            },
            "fingerprint": self.model.fingerprint(self.read_size, GROUP, self.device).tolist(),
            "fingerprint_config": {
                "img": self.read_size,
                "n_slot": self.model.n_slot,
                "group": GROUP,
                "seed": FINGERPRINT.seed,
            },
        }


@dataclass(frozen=True)
class StampedRun:
    """What a checkpoint says about the RUN that wrote it, read back out of `MemberRecord.payload`.

    Beside the writer because it is the same statement read in the other direction, and because two
    consumers had already started deriving it themselves: `metrics_recipe`, which moves the stamp into
    a scores file, and `holdout_backfill`, which needs the split a finished arm used.

    It exists at all because the stamp is the ONLY account most of the ledger kept. Of the 69 members
    on disk 50 wrote no metrics file, and 36 of those carry the full split recipe here — so a reader
    that looked only beside the checkpoint called three quarters of the ledger unreadable for want of
    a file rather than a fact.
    """

    options: dict[str, str]
    """The run's flags as strings, which is how `payload` had to store them: every loader here reads
    with `weights_only=True`, and that refuses to unpickle a Path."""

    config: dict[str, str]
    """The reading and architecture block, which records the FIELD OF VIEW for every era of the ledger.

    Separate from `options` because the two disagree about what exists: `options` is `vars(RunOptions)`
    at the time of the run, so `wsfold0_e6` — the whole-stack control — carries no `crop_mm` key at all,
    while its config records `crop_mm 130.0` because the saver derived it from a nested `crop` object.
    A reader that asks only `options` therefore cannot say what view the control read at, and has to
    treat every crop arm as un-judgeable rather than as the view change it plainly is."""

    kept_epoch: int | None
    """The epoch whose weights were SAVED — not the last one run. Under patience those differ, and
    predicting against the later number describes a model that was thrown away.

    None for the older half of the ledger, which wrote no such key. It is None rather than derived
    from `epochs_done`: that read is what stamped `wsfold0_e6`'s predictions as epoch 11 when its
    weights are the epoch-6 best. `epochs_done` is only the kept epoch plus one for a run that kept
    its LAST epoch — on all three pre-`epoch` checkpoints checked it equals `options.epochs`, the
    length the run was asked for, which under patience is a different epoch entirely."""

    @classmethod
    def of(cls, checkpoint: Path) -> "StampedRun | None":
        """The stamp, or None for a checkpoint written before runs recorded one."""
        if not checkpoint.exists():
            return None
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        options = saved.get("options")
        if not options:
            return None
        kept = saved.get("epoch")
        return cls(
            options={name: str(value) for name, value in options.items()},
            config={name: str(value) for name, value in (saved.get("config") or {}).items()},
            kept_epoch=None if kept is None else int(kept),
        )
