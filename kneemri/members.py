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
from typing import NotRequired, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from kneemri.backbone import ENCODERS, SMALL
from kneemri.model import StudyDesign, StudyModel
from kneemri.slots import SCHEMES

FINGERPRINT_TOLERANCE = 2e-3
"""Above GPU-to-GPU numeric noise (~1e-5), far below what any preprocessing change moves (order one)."""

CANONICAL = "canonical"
"""The scheme a member names when its slot list is the published six."""

LEGACY_SLICES = 12
"""What a manifest written before members could differ in depth was fitted on."""


class ReadingRules(TypedDict):
    """The four decisions that settle what a slice IS, none of which changes a tensor shape."""

    order: str
    lat: str
    slot_fallback: bool
    decode_fill: str


class MemberConfig(TypedDict):
    """The manifest's per-member config: the reading a member was fitted under, plus its architecture."""

    img: int
    witness: NotRequired[dict[str, object]]
    """Training-path predictions for a few named studies; absent on members predating the check."""

    whole_stack: NotRequired[bool]
    """Absent on every member fitted before whole-stack reading existed; those were windowed."""

    encoder_layer: NotRequired[int]
    """Which encoder block the head read; absent on every member fitted before it was a flag."""

    bf16: NotRequired[bool]
    """Whether the run's autocast was bfloat16; absent on every member fitted before the flag existed."""

    train_size: NotRequired[int]
    """Absent on every member fitted before the read size could differ from the cache's."""

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
class Window:
    """How a member is read at test time: where its windows begin, how wide, at what resolution.

    One object because the three are never chosen apart — a whole-stack member wants one centred
    window of full width, a windowed one wants three-slice windows, and the read size follows the
    member's own config either way. They were three positional arguments, which tripped the
    argument-count gate and, more to the point, let a caller vary one and forget the others.
    """

    starts: list[int]
    width: int
    size: int


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
    def read_size(self) -> int:
        """Pixels per side the model actually READ, which is the cache's size unless it resampled.

        Defaults to `image_size` when the key is absent, which is what every member fitted before
        resampling existed means — so the two whole-stack members already on disk keep verifying
        against their stored fingerprints rather than being refused by a field they predate.
        """
        return int(self.config.get("train_size") or self.config["img"])

    @property
    def whole_stack(self) -> bool:
        """Whether this member was fitted reading EVERY cached slice in one pass.

        Recorded in every checkpoint since whole-stack training landed, and read by nobody until
        2026-08-12 — the same defect as `slice_pooling`, which was also written faithfully and ignored.

        It changes what inference must do, not merely how it is described. A whole-stack member's slot
        embedding is the elementwise maximum over the FULL stack; serving it three slices at a time
        gives it maxima over a quarter of the stack, which is a different function of the same pixels.
        Nothing raises, the fingerprint passes (it fixes image size and group, not stack depth), and the
        holdout cannot see it because holdout scoring reads the same window training did.

        Defaults to False for every member fitted before the flag existed, including all twenty
        published ones — they were windowed, and windowed is what they must stay.
        """
        return bool(self.config.get("whole_stack", False))

    @property
    def layer(self) -> int:
        """Which encoder block this member read, defaulting to the last for everything fitted before."""
        return int(self.config.get("encoder_layer", -1))

    @property
    def precision(self) -> torch.dtype:
        """The autocast dtype this member was fitted under, which inference must match.

        `bf16` has been recorded in every checkpoint since the flag landed, with a comment saying that
        bf16 and fp16 are different arithmetic — and read by nobody, so `Ensemble.predict_member` ran a
        dtype-less `torch.autocast("cuda")`, whose default is FLOAT16. Every bf16 member was therefore
        served in fp16. The eighth instance here of a value travelling in a config while its consumer
        reads a constant, and the first found by the witness check rather than by a lost submission.

        It is not only a precision question. bfloat16 carries float32's exponent range and float16 does
        not, so an activation a bf16-trained model produces comfortably can overflow to inf under fp16 —
        on the T4 the kernel actually submits from, where there is no second chance.

        Defaults to float16, which is what every member fitted before the flag existed genuinely used.
        """
        return torch.bfloat16 if self.config.get("bf16", False) else torch.float16

    @property
    def witness(self) -> dict[str, object]:
        """What the training path predicted for a few named studies, or empty for members without one.

        The fingerprint proves the WEIGHTS are the weights. This proves the READING is the reading, and
        that is the gap four defects walked through in one session — whole-stack reading windowed to
        three slices, the decode resolution and the slot list each taken from a default. All four
        loaded cleanly and verified cleanly, because the fingerprint pins image size and group only.

        Empty for the twenty published members and for anything trained before this existed; a check
        that cannot run is not a failure, it is an unverified member, and it must say so rather than
        pass quietly.
        """
        return dict(self.config.get("witness", {}))

    @property
    def slot_count(self) -> int:
        """How many acquisition slots the head attends over."""
        return len(self.config["slots"])

    @property
    def group(self) -> int:
        """Consecutive slices stacked as the encoder's three channels."""
        return int(self.config["group"])

    @property
    def slices(self) -> int:
        """How deep a stack this member was fitted on — the depth inference has to decode.

        `slices` was in `MemberConfig` from the start with nothing to read it, exactly as `variant`
        was: the key existed, the accessor did not, and inference used a hardcoded twelve instead.
        Defaults to twelve for manifests written before members could differ in depth.
        """
        return int(self.config.get("slices", LEGACY_SLICES))

    @property
    def pool(self) -> str:
        """Which token-pooling scheme the encoder read-out uses."""
        return str(self.config["pool"])

    @property
    def lora_rank(self) -> int:
        """The rank of this member's adapters, or 0 if its encoder was fine-tuned directly.

        Zero for every manifest written before low-rank adaptation existed, which is correct: those
        members carry no adapter keys and must be built without them.
        """
        return int(str(self.config.get("lora_rank", 0) or 0))

    @property
    def slice_pooling(self) -> str:
        """How the member turns a group of slices into one feature — "channels" or "max".

        This is not a preference, it is a different FUNCTION on different pixels: "channels" stacks
        three slices as the encoder's RGB, "max" encodes each slice alone and takes the per-dimension
        maximum. A member fitted one way and run the other predicts from an image it never saw.

        It was recorded in every checkpoint and read by nobody — `load` built StudyDesign(pool=...) and
        let this fall back to the default. The fingerprint check would have refused such a member
        rather than mispredicting, so the failure was loud, but it made every max-pooling member we
        train unsubmittable. Defaults to "channels" for the published members, which predate the
        option and were all fitted that way.
        """
        return str(self.config.get("slice_pooling", "channels"))

    @property
    def variant(self) -> str:
        """Which DINOv2 size this member was fitted on, so the kernel mounts the matching weights.

        Defaults to small for any manifest written before variants existed — every member published so
        far is small, and guessing wrong here is not a loud failure: the widths differ, so it surfaces
        as a shape error during load if we are lucky and as a silently wrong model if not.
        """
        return str(self.config.get("variant", SMALL))

    def window(self, available: int, *, overlap: bool) -> Window:
        """How THIS member is read at test time, given a decoded stack of `available` slices.

        One definition, so every reader shares it. There were three: `Ensemble.predict_member` branched
        inline, `member_curve` grew its own `Reading`, and `HoldoutPredictions` took three loose
        arguments a caller could vary one at a time. Each place that decided independently got it wrong
        at least once — a whole-stack member windowed by three is a different function, and it reached
        the leaderboard that way (knee-y8y).

        A whole-stack member gets ONE centred window of its full depth: its slot embedding is the
        elementwise maximum over the whole stack, so there is nothing to overlap and nothing to average.
        A windowed member gets its group-sized windows, overlapping or disjoint as asked.
        """
        if self.whole_stack:
            width = min(self.slices, available)
            return Window(starts=[(available - width) // 2], width=width, size=self.read_size)
        starts = (
            list(range(available - self.group + 1))
            if overlap and available >= self.group
            else [index * self.group for index in range(max(available // self.group, 1))]
        )
        return Window(starts=starts, width=self.group, size=self.read_size)

    @classmethod
    def from_checkpoint(cls, path: Path) -> "Member":
        """A member read straight from a training checkpoint, without a package around it.

        Analysis reads the same fields packaging does — resolution, depth, pooling, scheme, whether the
        whole stack is read — and every tool that re-derived them from a raw checkpoint got at least one
        wrong. This is the same type inference uses, so there is one description of a member's reading
        and analysis cannot drift from it.
        """
        saved = torch.load(path, map_location="cpu", weights_only=True)
        holdout = float(saved.get("holdout", float("nan")))
        return cls(
            identifier=path.stem,
            file=path,
            fold=int(str(saved.get("options", {}).get("fold", 0))),
            config=saved.get("config", {}),
            fingerprint=np.asarray(saved.get("fingerprint", []), np.float32),
            holdout=holdout,
            annot=float(saved.get("annot", float("nan"))),
        )

    @property
    def scheme(self) -> str:
        """Which named slot scheme this member's recorded slot list is, so its cache can be found.

        On the member because it is a fact ABOUT the member, and because two tools had already grown
        their own copy of this lookup — the curve and the witness check — which is how a description
        drifts from the thing it describes.
        """
        known = ((name, [slot.name for slot in slots]) for name, slots in SCHEMES.items())
        return next((name for name, names in known if names == list(self.config["slots"])), CANONICAL)

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

    def strongest(self, count: int) -> list[Member]:
        """The `count` members with the best recorded holdout, best first.

        For the Efficiency Prize, where the accuracy leaderboard and the efficiency one reward opposite
        things: the best accuracy anywhere on the efficiency board sits at rank 638, while our accuracy
        run cheaply is worth ranks 29-50.

        HOW MANY TO ASK FOR is not a free choice, and the holdout answer this docstring used to give was
        wrong on the leaderboard. On a SEED ensemble's holdout, members two and three bought +0.0018 and
        +0.0007 and the best single outscored the three-member average, which reads as "one is enough".
        Scored entries say otherwise: one member 0.868, two whole-stack 0.882, three whole-stack 0.882,
        five windowed 0.888. Ask for two — one costs 0.020, and a third buys nothing.

        Ranking by each member's OWN holdout is the only ordering available at submission time, and it
        is a weak one: those numbers come from different folds and, for anything trained before the
        label table was recovered, from different rulers. It picks a good member, not provably the best.
        """
        return sorted(self.members, key=lambda member: member.holdout, reverse=True)[:count]

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
        # The member records BOTH its encoder and the size it read at, and nothing until now checked
        # that the two are compatible: DINOv2 tiles 392 and DINOv3 does not, so a member could name a
        # combination no patch grid can express and only fail inside a kernel.
        ENCODERS[member.variant].require_tiling(member.read_size)
        design = StudyDesign(pool=member.pool, slice_pooling=member.slice_pooling, layer=member.layer)
        model = StudyModel(backbone, dim, member.slot_count, design)
        # THE ADAPTERS MUST EXIST BEFORE THE WEIGHTS ARRIVE. A LoRA member's state dict carries adapter
        # keys that only exist once `low_rank` has wrapped the encoder's linears; building the plain
        # architecture and loading into it fails on unexpected keys at best, and at worst — under a
        # non-strict load — returns a model with the pretrained base and no adaptation at all, which
        # scores like a broken member for no visible reason.
        if member.lora_rank:
            model.low_rank(member.lora_rank)
        model = model.to(device)
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
        got = model.fingerprint(member.read_size, member.group, device)
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
