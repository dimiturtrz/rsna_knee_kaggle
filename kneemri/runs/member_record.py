"""What a trained member records about itself, so a checkpoint can say what produced it."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from core.data.corpus.crop_centre import Crop
from core.data.corpus.slot_reader import GROUP, Band
from core.data.corpus.slots import Routing
from core.findings import TARGETS
from kneemri.models.encoder import Encoder
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
    "config": "the recipe file; each value it set is recorded under its own field name",
    "resume": "recovery rather than recipe: it says where a killed run picked up, not what it fitted, "
    "and `epochs_done` already records how far the run got",
    "epochs": "run length — `epochs_done` records what was actually reached, beside the kept epoch",
    "patience": "when the run stopped, which the kept epoch already states",
    "stop_after_epoch": "where a round ended its loop — the same write-off as `patience` with a "
    "deterministic trigger instead of a holdout one. The SCHEDULE is built for `epochs` whatever this "
    "says, so a rebuild reconstructs the same function; `epochs_done` records how far the loop got",
    "patience_from_progress": "when keeping could begin, which the kept epoch likewise already states",
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
    "exclude_fold": "which fold was withheld from TRAINING too — a property of the corpus a run was "
    "fitted on, and the recipe block carries it; a rebuild reconstructs the function, not the split",
    "pseudo_fold": "which fold was trained on AS WELL AS held out — the transductive cut, and a "
    "property of the corpus a run was fitted on exactly as `exclude_fold` is. A rebuild reconstructs "
    "the function, not the split",
    "one_cycle": "a schedule: it changes how the weights were reached, never how they are rebuilt",
    "init": "where the walk STARTED — the second stage of an LP-FT pair loads a member instead of a "
    "fresh head, which changes nothing about the module a rebuild constructs and everything about "
    "where its parameters came to rest. Those parameters ARE the checkpoint. The recipe block carries "
    "the path, which is where a two-stage arm's provenance is read from, and the run's log opens with "
    "the warm-start line `WarmStart.applied` returns",
    "init_scope": "how much of that starting checkpoint was taken — a parameter OF the write-off above, "
    "and dead without it. It moves where the parameters came to rest, never the module a rebuild "
    "constructs: a trunk-only start leaves a FRESH head, which is the same head shape this run's own "
    "flags describe. The recipe block carries it beside the path",
    "certainty_ramp": "a curriculum over the LABEL WEIGHTS, which live in the training table rather than "
    "in the member: it changes how fast the run came to trust each cell and leaves nothing behind to "
    "rebuild, exactly as the schedule write-off above does. The recipe block carries it, which is where "
    "an arm's curriculum is read from",
    "warmup": "the share of that schedule spent climbing — a parameter OF the write-off above, and dead without it",
    "weight_decay": "an optimiser setting: it shapes where the walk settled, and the weights that "
    "settled are what a rebuild loads. Adds no module and no parameter",
    "grad_clip": "same path and same write-off — a ceiling on how far one step could move, applied "
    "between backward and step, with nothing left in the checkpoint to rebuild",
    "sharpness": "same path and same write-off as `grad_clip` — a radius the step climbed before it "
    "descended, spent entirely between backward and step. It moves WHERE the walk settled and leaves "
    "no module and no parameter behind, so the member a rebuild constructs is byte-identical",
    "stochastic_depth": "same path and same write-off as `sharpness` — a residual branch dropped per "
    "sample during training only. `DropPath` holds no parameter and registers no buffer, and is identity "
    "in eval(), so a member trained under it rebuilds byte-identical and serves unaffected",
    "weight_floor": "supervision policy, same",
    "calibrated_targets": "supervision policy, same",
    "finding_calibration": "supervision policy, same — it shifts each finding's level in the training "
    "TABLE before the loss ever sees it, exactly as `calibrated_targets` (which it requires) does, and "
    "adds no parameter, so the head a rebuild serves is byte-identical",
    "severity_weight": "supervision policy, same — it grades the loss weight and leaves the head alone",
    "hold_annotations": "supervision policy, same — it decides which cells the loss READ and touches no "
    "parameter, so the head a rebuild serves is byte-identical",
    "hold_annotation_studies": "supervision policy, same — it zeroes the loss WEIGHT on 58 studies and "
    "touches no parameter, so the head a rebuild serves is byte-identical. It changes what the member "
    "may be READ on rather than what it serves: this is the only flag under which an annotated-read "
    "score is out of sample",
    "grade_ladder": "objective policy — the thresholds are subtracted INSIDE the loss, so the head still "
    "emits the one per-label logit a rebuild serves",
    "side_weight": "objective policy — the term compares two of the twelve logits against each "
    "other INSIDE the loss and adds no parameter, so the head a rebuild serves is byte-identical",
    "focus_label": "objective policy — it SCALES the unnamed columns of the loss WEIGHT and touches no "
    "parameter, so the twelve-logit head a rebuild serves is byte-identical. The hazard depends on the "
    "strength: at unfocused_weight 0 it is the MASK, whose unnamed columns are untrained, so such a "
    "member is an INSTRUMENT and never a roster seat; at a weight inside (0, 1) every column is still "
    "supervised and the member is shippable like any other",
    "unfocused_weight": "objective policy — the strength focus_label leans at, read INSIDE the loss and "
    "adding no parameter. Recorded beside the findings because a mechanism that lands at the wrong "
    "strength passes every gate: an unfocused weight applied twice descends its square",
    "sample_spread": "a draw policy — it decides WHICH studies the epoch descended on and touches no "
    "module, no parameter and no loss term, so the function a rebuild serves is byte-identical. The "
    "file it names is a fact about how these weights were REACHED, like `teacher_targets`, and the "
    "witness that it landed is the trainer's own `sampling_tilt`, not the checkpoint",
    "sample_alpha": "the tilt of that same draw policy, same write-off",
    "sample_ramp": "WHEN that same tilt arrives rather than whether it does — it changes the order the "
    "epochs saw the pool in and touches no module, no parameter and no loss term, so the write-off is "
    "`sample_spread`'s exactly. Its own witness is the trainer's `sampling_ramp`, kept separate from "
    "`sampling_tilt` precisely because a ramp that failed to attach would otherwise report the static "
    "arm's tilt and be indistinguishable from it",
    "corpus_oversample": "a draw policy — it decides WHICH studies the epoch descends on and touches no "
    "module, no parameter and no loss term, so the function a rebuild serves is byte-identical. The "
    "corpus tag it weights is a fact about how these weights were REACHED, not part of the function",
    "target_corpus": "the companion axis of that same draw policy: which tag is treated as the target "
    "domain changes the epoch order and nothing else, so it is written off with `corpus_oversample`",
    "scale_jitter": "an input augmentation applied while fitting, never while serving — it widens the "
    "field-of-view jitter each training study is resampled under, and a rebuild serves the cache's own "
    "crop, so the function is byte-identical for the same reason `intensity_aug` is",
    "padding_mode": "the out-of-frame rule of that same fitting-time resample — it only has an effect "
    "where the sampling grid leaves the cached crop, which never happens while serving, so a rebuild "
    "serves a byte-identical function for the same reason `scale_jitter` does",
    "ghosting": "an input augmentation applied while fitting, never while serving - it adds displaced "
    "replicas of the training image to itself at a severity measured off KMAR-50K's paired volumes, and "
    "a rebuild serves the unghosted image, so the function is byte-identical for the same reason "
    "`intensity_aug` is",
    "aug_start": "a progress schedule over fitting-time regularisers only — augmentation, mixup and "
    "stochastic depth are all identity while serving, so a rebuild serves a byte-identical function; the "
    "run's recipe block carries the five knobs",
    "aug_mid": "the plateau of that same fitting-time schedule, same write-off as `aug_start`",
    "aug_end": "the closing strength of that same fitting-time schedule, same write-off as `aug_start`",
    "aug_ramp": "where that same fitting-time schedule's plateau begins, same write-off as `aug_start`",
    "aug_tail": "where that same fitting-time schedule's plateau ends, same write-off as `aug_start`",
    "hard_start": "a progress schedule over EXAMPLE HARDNESS — label trust and the draw's tilt, both of which "
    "live in the training table and the sampler rather than in the member, the same write-off as "
    "`certainty_ramp`; the run's recipe block carries the five knobs",
    "hard_mid": "the plateau of that same hardness schedule, same write-off as `hard_start`",
    "hard_end": "the closing hardness of that same schedule, same write-off as `hard_start`",
    "hard_ramp": "where that same hardness schedule's plateau begins, same write-off as `hard_start`",
    "hard_tail": "where that same hardness schedule's plateau ends, same write-off as `hard_start`",
    "slot_dropout": "an input augmentation applied while fitting, never while serving — it drops whole "
    "SLOTS from the six-slot list during training, and a rebuild serves the full list, so the function "
    "is byte-identical for the same reason `intensity_aug` is",
    "patch_erase": "an input augmentation applied while fitting, never while serving — it blanks one grid "
    "cell of a study during training, and a rebuild serves the unblanked image, so the function is "
    "byte-identical for the same reason `slot_dropout` is",
    "patch_maps": "the occlusion sweep `patch_erase` draws that cell from — a fact about which cells the "
    "epochs saw blanked, like `sample_spread`, and the witness that it landed is the run's own recipe "
    "block, never the checkpoint",
    "label_errors": "a SUPERVISION policy — it zeroes the loss weight of the cells a flag table names "
    "and touches no module, no parameter and no loss term, so the twelve-logit head a rebuild serves is "
    "byte-identical. The file it names is a fact about which cells the epochs descended on, like "
    "`sample_spread`, and the witness that it landed is the run's own recipe block plus the flag "
    "table's row count, never the checkpoint",
    "unread_targets": "the companion supervision policy to `label_errors`, and written off for the same "
    "reason from the other side: that one zeroes the WEIGHT of a cell the reader got wrong, this one "
    "replaces the SCORE of a cell the reader never read. Neither adds a module, a parameter or a loss "
    "term, so the twelve-logit head a rebuild serves is byte-identical; the file it names is a fact "
    "about what the epochs descended toward, and the witness that it landed is the run's recipe block "
    "plus the table's row count, never the checkpoint",
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
    "contrast_weight": "objective policy — it shapes the head's per-finding context vector inside the "
    "loss, and that vector is already computed on every forward pass; the term adds no parameter and "
    "the head a rebuild serves is byte-identical",
    "dissent_weight": "objective policy — it re-weights the mass of the SAME label-ordering term by how "
    "badly a teacher pool orders each pair, inside the loss, adding no parameter; the head a rebuild "
    "serves is byte-identical",
    "feature_weight": "objective policy — it pulls the head's per-finding context toward a teacher's "
    "representation through a projection that lives on the LOSS and is thrown away with it, so the "
    "twelve logits a rebuild serves come out of the same head, byte-identical",
    "teacher_targets": "the ensemble that taught it, which is a fact about how these weights were "
    "REACHED — a rebuild loads the student, and the teacher is not part of the function it is",
    "teacher_features": "the representations that taught it — the same fact about how the weights were "
    "reached, one level below the verdict, and equally not part of the function they are",
    "encoder_weights": "where the encoder STARTED, which the fitted weights have already absorbed — a "
    "rebuild loads this member, not the initialisation it grew out of",
    "encoder_adapt_depth": "how much of that starting point was taken — a fact about the run's history, "
    "not about the function these weights are, and a rebuild loads the fitted encoder whole",
    "mixup": "an input-AND-target policy applied while fitting, never while serving — it blends the "
    "batch with a permutation of itself and adds no parameter, so the head a rebuild serves is "
    "byte-identical. It is `intensity_aug`'s write-off with the target moved too; the recipe block "
    "carries the alpha, which is where the landed-check reads it",
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

SPELLED HERE rather than imported from `TrainConfig`, for the reason `RunStamp` is a value:
train.py imports this module, so naming its dataclass would close the cycle.
"""


@dataclass(frozen=True)
class RunStamp:
    """The run settings a checkpoint has to record, typed, and nothing a rebuild does not need.

    A value rather than the config tree itself, so this module does not import the CLI that produces it:
    train.py imports this one, and serving reads `StampedRun` from here without ever touching training.
    `TrainConfig.stamp` is the one place that fills it, so a knob a rebuild must know is added there
    and here together or the constructor refuses. It also lets a test build one without a command line.
    """

    recipe: dict[str, str]
    image_size: int
    crop: Crop
    band: Band
    spacing: str
    routing: Routing
    train_size: int
    cache_studies: int
    slices: int
    pool: str
    slice_pooling: str
    slice_regions: int
    slot_pooling: str
    encoder_layer: int
    fuse_blocks: int
    slice_shift: int
    through_plane: bool
    anatomical_prior: bool
    lora_rank: int
    report_weight: float
    whole_stack: bool
    unfreeze_last: int
    unfreeze_offset: int
    unfreeze_stem: bool
    head_lr: float
    backbone_lr: float
    layer_decay: float
    ema: bool
    bf16: bool
    rank_weight: float
    pair_memory: int


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

    options: RunStamp
    encoder: Encoder
    model: StudyModel
    device: torch.device
    slot_names: list[str]
    run: dict[str, object] | None = None
    """`RunIdentity.stamp` — which run these weights came from; None where no run was begun."""

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
            "run": self.run,
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
                # THE RUN'S OWN SPACING, on the axis the band does not cover: the band says how much
                # depth is visible, this says whether consecutive samples are consecutive in
                # millimetres. The serve path reads it to build the reader.
                "spacing": options.spacing,
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
                "fuse_blocks": options.fuse_blocks,
                "slice_shift": options.slice_shift,
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
                # WHERE that many blocks sat. The count alone cannot tell an early-block fit from the
                # suffix every run before 2026-09-05 opened, and the loader has to rebuild the recipe.
                "unfreeze_offset": options.unfreeze_offset,
                # WHETHER THE PIXEL-READING LAYERS MOVED, which neither number above can express: the
                # trainable set is built out of blocks, so the stem is frozen at every depth unless this
                # is set, and a rebuild that defaulted it would fit a different trunk (knee-p6fr8).
                "unfreeze_stem": options.unfreeze_stem,
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

    Separate from `options` because the two disagree about what exists: `options` is `RunStamp.recipe`
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

    run_id: str | None = None
    """The id `RunIdentity` minted for the run, None for a checkpoint written before runs had one."""

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
        run = saved.get("run") or {}
        return cls(
            options={name: str(value) for name, value in options.items()},
            config={name: str(value) for name, value in (saved.get("config") or {}).items()},
            kept_epoch=None if kept is None else int(kept),
            run_id=str(run["id"]) if "id" in run else None,
        )
