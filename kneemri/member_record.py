"""What a trained member records about itself, so a checkpoint can say what produced it."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from kneemri.backbone import Encoder
from kneemri.model import TARGETS, StudyModel
from kneemri.slot_reader import CROP_MM, SLICE_BAND


class RunOptions(Protocol):
    """The run settings a checkpoint has to record, as a shape rather than a named class.

    Structural, so this module does not import the CLI that produces it — train.py imports this one,
    and a concrete type here would make that a cycle. It also lets a test construct the fields without
    a parsed command line.
    """

    @property
    def image_size(self) -> int: ...

    @property
    def encoder_layer(self) -> int: ...

    @property
    def train_size(self) -> int: ...

    @property
    def slices(self) -> int: ...

    @property
    def slice_pooling(self) -> str: ...

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
    def report_weight(self) -> float: ...

    @property
    def whole_stack(self) -> bool: ...

    @property
    def unfreeze_last(self) -> int: ...

    @property
    def head_lr(self) -> float: ...

    @property
    def backbone_lr(self) -> float: ...


GROUP = 3
FINGERPRINT_SEED = 2026

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

    def witness(
        self,
        identifiers: list[str],
        holdout: NDArray[np.int64],
        predictions: NDArray[np.float64],
    ) -> dict[str, object]:
        """What the TRAINING path predicted for a few named holdout studies.

        Named studies rather than positions, because a position means nothing without the same corpus
        index — and the whole point is to be checkable from a different process, against a cache built
        later, by code that never saw this run.
        """
        chosen = holdout[:WITNESS_STUDIES]
        return {
            "studies": [identifiers[position] for position in chosen],
            "predictions": predictions[: len(chosen)].tolist(),
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
            "options": {name: str(value) for name, value in vars(options).items()},
            "targets": list(TARGETS),
            "slots": list(self.slot_names),
            "config": {
                # Inside the config so it reaches the manifest for free: package_members copies
                # config wholesale, and a witness that does not travel with the member cannot check it.
                "witness": witness or {},
                "img": options.image_size,
                "train_size": self.read_size,
                "slices": options.slices,
                "group": GROUP,
                "crop_mm": CROP_MM,
                "band": list(SLICE_BAND),
                "slots": list(self.slot_names),
                "rules": {"order": "normal", "lat": "centre", "slot_fallback": False, "decode_fill": "nearest"},
                "variant": self.encoder.variant,
                "pool": "cls_mean",
                "slice_pooling": options.slice_pooling,
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
                "report_weight": options.report_weight,
                "whole_stack": options.whole_stack,
                # The value this run USED, never the module constant. Recording the constant made the
                # field a lie the moment the depth became a flag, and the loader reads it back to
                # decide how a member was fitted.
                "unfreeze_last": options.unfreeze_last,
                "head_lr": options.head_lr,
                "backbone_lr": options.backbone_lr,
            },
            "fingerprint": self.model.fingerprint(self.read_size, GROUP, self.device).tolist(),
            "fingerprint_config": {
                "img": self.read_size,
                "n_slot": self.model.n_slot,
                "group": GROUP,
                "seed": FINGERPRINT_SEED,
            },
        }
