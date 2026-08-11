"""Reading a package of trained members, and proving each one is the model it was.

A member is only reproducible together with the reading that produced it — resolution, crop, slice
order, laterality rule, slot predicate — and none of those change a single tensor shape. So a member
fitted under one reading and run under another loads cleanly, runs, and returns a submission computed
from the wrong image. Nothing raises.

The defence is `StudyModel.fingerprint`: each member carries the answer it gave to a seeded question
with no data in it. Any preprocessing difference moves that answer by order one; two GPUs disagreeing
move it by about 1e-5, which is why the tolerance sits between them rather than at zero.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from kneemri.model import StudyDesign, StudyModel

FINGERPRINT_TOLERANCE = 2e-3
"""Above GPU-to-GPU numeric noise (~1e-5), far below what any preprocessing change moves (order one)."""


class ReadingRules(TypedDict):
    """The four decisions that settle what a slice IS, none of which changes a tensor shape."""

    order: str
    lat: str
    slot_fallback: bool
    decode_fill: str


class MemberConfig(TypedDict):
    """The manifest's per-member config: the reading a member was fitted under, plus its architecture."""

    img: int
    slices: int
    group: int
    crop_mm: float
    band: list[float]
    slots: list[str]
    rules: ReadingRules
    variant: str
    pool: str


class WeightsError(RuntimeError):
    """Raised when attached weights cannot be trusted to be the ones that were fitted.

    Deliberately fatal: a run that predicts from a mismatched model completes, writes a plausible
    submission, and differs from a correct one only in a number no output of the run reveals.
    """


@dataclass(frozen=True)
class Member:
    """One trained model in a package, together with the reading it was fitted under."""

    identifier: str
    file: Path
    fold: int
    config: MemberConfig
    fingerprint: NDArray[np.float32]
    holdout: float
    annot: float

    @property
    def image_size(self) -> int:
        """Pixels per side the encoder is fed — with the crop, this fixes the effective mm/pixel."""
        return int(self.config["img"])

    @property
    def slot_count(self) -> int:
        """How many acquisition slots the head attends over."""
        return len(self.config["slots"])

    @property
    def group(self) -> int:
        """Consecutive slices stacked as the encoder's three channels."""
        return int(self.config["group"])

    @property
    def pool(self) -> str:
        """Which token-pooling scheme the encoder read-out uses."""
        return str(self.config["pool"])

    @property
    def pixel_group(self) -> str:
        """Members sharing this key need identical pixels, so one decode serves all of them."""
        pixels = {
            "band": self.config["band"],
            "crop_mm": self.config["crop_mm"],
            "group": self.config["group"],
            "img": self.config["img"],
            "rules": self.config["rules"],
            "slices": self.config["slices"],
            "slots": self.config["slots"],
        }
        return json.dumps(pixels, sort_keys=True)


class MemberPackage:
    """A mounted weights package: the manifest, its members, and the check that they are trustworthy."""

    def __init__(self, root: Path) -> None:
        self.root = root
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.members = [
            Member(
                identifier=entry["id"],
                file=root / entry["file"],
                fold=int(entry["fold"]),
                config=entry["config"],
                fingerprint=np.asarray(entry["fingerprint"], np.float32),
                holdout=float(entry["holdout"]),
                annot=float(entry["annot"]),
            )
            for entry in manifest["members"]
        ]
        missing = [member.file.name for member in self.members if not member.file.is_file()]
        if missing:
            raise WeightsError(
                f"{root} holds a manifest listing {len(self.members)} members but "
                f"{len(missing)} of their files are absent (first {missing[0]!r})"
            )

    @classmethod
    def find(cls, base: Path = Path("/kaggle/input"), name: str = "manifest.json") -> "MemberPackage | None":
        """Locate a mounted package, or return None when none is attached.

        The manifest decides, not the filenames beside it: testing for a naming convention makes the
        search agree with whatever the packager last called its files, which is a second definition of
        what a package is. The competition's own image directories are skipped — walking them is
        hundreds of thousands of stat calls for a file that is not there.
        """
        if not base.is_dir():
            return None
        for candidate in sorted(base.rglob(name)):
            if {"train_series", "test_series"} & set(candidate.parts):
                continue
            return cls(candidate.parent)
        return None

    def load(self, member: Member, backbone: nn.Module, dim: int, device: torch.device) -> StudyModel:
        """Build the member's model, load its weights, and refuse it if the fingerprint disagrees."""
        model = StudyModel(backbone, dim, member.slot_count, StudyDesign(pool=member.pool)).to(device)
        model.load_trained(self.state_dict_of(member, device))
        self.check(model, member, device)
        return model

    def state_dict_of(self, member: Member, device: torch.device) -> dict[str, Tensor]:
        """The member's tensors, unwrapped from whatever the checkpoint wraps them in.

        A published checkpoint is a training artefact, not a bare state dict: it carries the weights
        under `model` alongside the epoch, the fold scores, the config, the out-of-fold predictions and
        its own copy of the fingerprint. Loading the wrapper directly fails loudly on unexpected keys —
        which is the good case, and the reason to unwrap explicitly rather than to load non-strictly.
        """
        checkpoint = torch.load(member.file, map_location=device, weights_only=True)
        return checkpoint.get("model", checkpoint)

    def check(self, model: StudyModel, member: Member, device: torch.device) -> float:
        """Compare the model's answer to the seeded question against the stored one.

        Returns the largest absolute difference, so a caller can report how much slack was left.
        """
        got = model.fingerprint(member.image_size, member.group, device)
        if got.shape != member.fingerprint.shape:
            raise WeightsError(
                f"{member.identifier}: fingerprint shape {got.shape} != stored "
                f"{member.fingerprint.shape}: the architecture is not the one these weights were fitted to"
            )
        difference = float(np.abs(got - member.fingerprint).max())
        if difference > FINGERPRINT_TOLERANCE:
            raise WeightsError(
                f"{member.identifier}: fingerprint differs by {difference:.4g} (tolerance "
                f"{FINGERPRINT_TOLERANCE:g}). The weights load but do not compute what they computed "
                f"when fitted — preprocessing, resolution or architecture has moved between the two runs."
            )
        return difference
