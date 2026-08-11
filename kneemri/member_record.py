"""What a trained member records about itself, so a checkpoint can say what produced it."""

from dataclasses import dataclass
from typing import Protocol

import torch

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
    def slices(self) -> int: ...

    @property
    def slice_pooling(self) -> str: ...

    @property
    def anatomical_prior(self) -> bool: ...

    @property
    def ema(self) -> bool: ...

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

    def payload(self, state: dict[str, torch.Tensor], holdout: float, epoch: int) -> dict[str, object]:
        """The full checkpoint for a set of weights — identical in shape whenever it is written."""
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
                "img": options.image_size,
                "slices": options.slices,
                "group": GROUP,
                "crop_mm": CROP_MM,
                "band": list(SLICE_BAND),
                "slots": list(self.slot_names),
                "rules": {"order": "normal", "lat": "centre", "slot_fallback": False, "decode_fill": "nearest"},
                "variant": self.encoder.variant,
                "pool": "cls_mean",
                "slice_pooling": options.slice_pooling,
                "anatomical_prior": options.anatomical_prior,
                "ema": options.ema,
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
            "fingerprint": self.model.fingerprint(options.image_size, GROUP, self.device).tolist(),
            "fingerprint_config": {
                "img": options.image_size,
                "n_slot": self.model.n_slot,
                "group": GROUP,
                "seed": FINGERPRINT_SEED,
            },
        }
