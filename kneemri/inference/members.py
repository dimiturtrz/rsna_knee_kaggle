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
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from core.data.corpus.crop_centre import CROP_MM, DEFAULT_CENTRE, Crop
from core.data.corpus.slot_reader import SPREAD, Band, Reading
from core.data.corpus.slots import CANONICAL_SCHEME, FIRST_MAXIMAL, SCHEMES, Routing
from core.data.corpus.study_cache import StudyCache
from core.mounts import Mounts
from core.paths import DataRoot
from kneemri.models.backbone import ENCODERS, SMALL
from kneemri.models.model import StudyDesign, StudyModel

logger = logging.getLogger(__name__)

FINGERPRINT_TOLERANCE = 2e-3
"""Above GPU-to-GPU numeric noise (~1e-5), far below what any preprocessing change moves (order one)."""

LEGACY_SLICES = 12
"""What a manifest written before members could differ in depth was fitted on."""


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

    cache_studies: NotRequired[int]
    """How many studies the cache it was fitted on held; absent means the full corpus, as every member
    fitted before a truncated cache could exist."""

    slices: int
    group: int
    crop_mm: float

    tie: NotRequired[str]
    """Which of several equally dense candidates filled a slot. Absent on every member fitted before the
    rule could vary, and those were all fitted under `first`."""

    centre: NotRequired[str]
    """Where the physical crop was taken about. Absent means the image centre, as every member on disk."""
    band: list[float]
    slots: list[str]
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
    def routing(self) -> Routing:
        """How this member's series became slots: which slots, and which candidate won a tie."""
        return Routing(scheme=self.scheme, tie=self.tie)

    @property
    def tie(self) -> str:
        """Which of several equally dense candidates filled a slot when this member was fitted.

        MEASURED, not assumed: the twenty published members reproduce their own recorded predictions
        better under LAST-maximal than under the first-maximal rule our `max()` applies — 4 of 4 members
        tested across 3 seeds, mean residual 15-25% lower (knee-evn). Ours were fitted under `first`, so
        this cannot be a global switch: one flag would fix the graft and break everything of ours.

        Defaults to `first`, which is what every cache and every member of ours on disk was built with.
        """
        return str(self.config.get("tie", FIRST_MAXIMAL))

    @property
    def band(self) -> Band:
        """The fraction of the stack this member's slices were sampled from, as a pair.

        Recorded since the first manifest and read by the decode since none of them: the reader took its
        module default, so a member fitted on a wider band would have been served twelve slices drawn
        from a narrower one — the same silent train/inference mismatch the crop had, on the other axis.
        A list on disk because JSON has no tuples; a `Band` here because it is a reading rule — it
        names the cache it produces, it is a grouping key, and the decode builds its reader from it.
        """
        return Band.of(self.config.get("band", Band().recorded()))

    @property
    def crop(self) -> Crop:
        """The physical crop this member was fitted under — its extent, and the point it was taken about.

        Both were recorded in every manifest and read by NOTHING until 2026-08-14: the decode built its
        reader from module constants, so a member fitted on a different crop was served under 130 mm
        about the image centre regardless, and nothing would have said so — every fingerprint passes
        against its OWN reading, and the witness replays the training path.

        Defaults are the historic behaviour, which is what every member on disk was fitted under and
        what its manifest means without saying so.
        """
        return Crop(
            mm=float(self.config.get("crop_mm", CROP_MM)),
            centre=str(self.config.get("centre", DEFAULT_CENTRE)),
        )

    @property
    def reading(self) -> Reading:
        """The geometry this member was fitted under, in the shape `StudyCache` takes it.

        The crop and the band were each recorded, each read back, and then each dropped on the way to
        the cache: every member-driven load named its cache by size, depth and router alone, so it
        resolved to the DEFAULT arrays whatever the member says it read. The build side could not
        clobber the historic cache once the key carried these two; the serve side could still be handed
        it, which is the same mismatch arriving from the other direction.

        One property rather than two arguments at four call sites, because the pair is what names a
        cache and a caller that remembers one and forgets the other is the defect itself.
        """
        return Reading(crop=self.crop, band=self.band, spacing=self.spacing)

    @property
    def spacing(self) -> str:
        """How this member's slices sat INSIDE its band — the historic spread unless it says otherwise.

        Defaulted for the reason `crop` gives: every manifest written before this axis existed means
        `SPREAD` without saying so, and it is the only value those members were ever fitted under.
        """
        return str(self.config.get("spacing", SPREAD))

    def cache(self, directory: Path) -> StudyCache:
        """The cache holding the pixels this member was fitted on, in a given data directory.

        THREE call sites spelled this out by hand — the oracle, the witness check and the curve — and
        each named the cache by size, depth and router while the member also records a crop and a band.
        So any member that read anything else resolved to the DEFAULT arrays, which exist, load, and are
        the wrong pixels: the mismatch the cache key was extended to prevent, arriving from the serve
        side instead of the build side.

        A method on the member because the set of fields that names a cache is not knowledge a consumer
        should hold: the next axis is added here once, rather than remembered in three places by whoever
        adds it. The axis LIST itself lives in `StudyCache.of`, because the train side had unified the
        same constructor for itself and the two unifications never met.
        """
        return StudyCache.of(self, directory)

    @property
    def cache_studies(self) -> int:
        """How many studies the cache this member was fitted on holds; 0 is the full corpus.

        A smoke member is fitted on a truncated cache, which differs from the full one in its first
        axis and in its index — the same pixels-per-position mismatch the crop and the band were added
        here to prevent, on the one axis that changes WHICH studies a position names. Absent from every
        manifest written before truncation existed, and those all read the full corpus.
        """
        return int(self.config.get("cache_studies", 0))

    @property
    def read_size(self) -> int:
        """Pixels per side the model actually READ, which is the cache's size unless it resampled.

        Defaults to `image_size` when the key is absent, which is what every member fitted before
        resampling existed means — so the two whole-stack members already on disk keep verifying
        against their stored fingerprints rather than being refused by a field they predate.
        """
        return int(self.config.get("train_size") or self.config["img"])

    def resized(self, size: int) -> "Member":
        """The same member READ at a different size — for pricing a resolution, never for predicting.

        A ViT interpolates its position embeddings, so this member will forward the bigger tensor and
        return a number that means nothing. What it spends getting there IS what a member fitted at
        that size spends, which is how a resolution arm is priced before it exists (knee-d2fc).

        THE SIZE HAS TO BE THE MEMBER'S, not the caller's tensor. `StudyModel.forward` resamples its
        input to the `img_size` it is handed and that comes from `window(...).size` — feeding it 448 px
        pixels while the member still says 392 measures a bilinear resize and an encoder running at the
        old resolution, which times the same as not overriding anything at all. Measured 2026-08-17:
        392, 420 and 448 all came back at 380 ms, and the flat line was the bug.
        """
        return replace(self, config={**self.config, "train_size": size})

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
        return StudyDesign.from_config(self.config).layer

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
        return StudyDesign.from_config(self.config).pool

    @property
    def lora_rank(self) -> int:
        """The rank of this member's adapters, or 0 if its encoder was fine-tuned directly.

        Zero for every manifest written before low-rank adaptation existed, which is correct: those
        members carry no adapter keys and must be built without them.
        """
        return StudyDesign.lora_rank_of(self.config)

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
        return StudyDesign.from_config(self.config).slice_pooling

    @property
    def slot_pooling(self) -> str:
        """How the twelve diagnoses read the six slots — "attend" or the ablation's uniform "mean".

        Wired to its consumers BEFORE the arm that uses it was queued, which is the whole point of
        writing it down here. A head fitted with uniform pooling and rebuilt with per-diagnosis
        attention loads without complaint and predicts from a different function; that shape has cost
        this project nine defects, and each was found after a run rather than before one.

        Defaults to "attend" — every member on disk was fitted that way, so the ledger's meaning is
        untouched.
        """
        return StudyDesign.from_config(self.config).slot_pooling

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
        # THE SLOT LIST LIVES IN TWO PLACES AND ONLY ONE OF THEM IS OURS. Our own checkpoints record it
        # inside `config`; the published ones record it at the TOP LEVEL and their config has no `slots`
        # key at all, so `scheme` and `slot_count` raised KeyError on every published checkpoint. That
        # went unnoticed because published members normally arrive through a manifest, which supplies
        # it - and the first tool to read one of their .pt files directly was `oof_oracle`, whose whole
        # purpose is to compare their pipeline against ours.
        config: MemberConfig = dict(saved.get("config", {}))  # type: ignore[assignment]
        if "slots" not in config and saved.get("slots"):
            config["slots"] = list(saved["slots"])
        return cls(
            identifier=path.stem,
            file=path,
            fold=int(str(saved.get("options", {}).get("fold", 0))),
            config=config,
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
        return next((name for name, names in known if names == list(self.config["slots"])), CANONICAL_SCHEME)

    @property
    def pixel_group(self) -> str:
        """Members sharing this key need identical pixels, so one decode serves all of them."""
        pixels = {
            "band": self.band.recorded(),
            # Adjacent and spread sample the SAME band at the same depth and return different
            # millimetres, so two members that differ here cannot share one decode.
            "spacing": self.spacing,
            # Extent AND centre: two members cropped differently are looking at different pixels, so one
            # decode cannot serve both.
            "crop": [self.crop.mm, self.crop.centre],
            # A different router puts different SERIES in a slot, which is different pixels.
            "tie": self.tie,
            "group": self.config["group"],
            # THROUGH THE ACCESSORS, because the group must be a function of the reading this member is
            # SERVED under. `slices` carries a default for manifests written before members could differ
            # in depth, so indexing the config here answered KeyError where every other consumer answers
            # twelve — one fact with two answers, and the one that decides which decode a member shares.
            "img": self.image_size,
            "slices": self.slices,
            "slots": self.config["slots"],
        }
        return json.dumps(pixels, sort_keys=True)

    @property
    def decode_key(self) -> tuple[str, int, str, Reading]:
        """Members sharing this key share ONE decode of the corpus, whatever size each then wants.

        NARROWER THAN `pixel_group` BY EXACTLY ONE AXIS, and that axis is the whole point.
        `pixel_group` is the unit of PREDICTION: a member is fed the cache size it was fitted from, so
        two resolutions are two sets of pixels to a model. Decoding is upstream of that —
        `SlotReader.cropped` stops before the resize and `Study.bags` returns every requested size from
        one cropped stack, each byte-identical to a native read — so resolution costs no read at all.

        It lived inline in `Ensemble.reads`, where nothing outside the serve path could ask the
        question. That mattered on 2026-08-29: the answer is that thirty-odd checkpoints we already own
        ride the decode our champion already pays for, and finding it out required building a throwaway
        package and dry-running it, because the key was only reachable through an `Ensemble`. Here, any
        tool holding checkpoints can group them, and — because it is the SAME property the serve path
        keys on — a tool's answer cannot drift from what a kernel will actually do.
        """
        return (self.scheme, self.slices, self.tie, self.reading)


SHIPPED_PACKAGE = "all_ours"
"""The directory under the processed root that `package_members` writes and `push.sh` uploads."""


class MemberPackage:
    """A mounted weights package: the manifest, its members, and the check that they are trustworthy."""

    @staticmethod
    def shipped_root(paths: Path = Path("paths.yaml")) -> Path:
        """Where the package that actually ships lives — the ONE directory allowed to carry its dataset id.

        Five CLIs defaulted `--root` to `comp/members`, an in-repo mirror of the same package, and
        `package_members --out` defaulted there too. Both directories stamped `dimiturnt/kneemri-members`
        while only the other one was pushed, so the mirror silently aged: on 2026-08-18 it held seven
        members against the live package's nine, and `mount_check`'s DEFAULT invocation refused a valid
        kernel because the member it names was absent from the copy nobody uploads. A check whose default
        answer is wrong teaches its operator to pass `--root` and stop reading the refusal.

        The weights are gitignored, so the mirror was never version control's copy of anything — it was a
        second home for a thing that can only have one.
        """
        return DataRoot.from_config(paths).processed() / SHIPPED_PACKAGE

    @classmethod
    def at(cls, root: Path | None, paths: Path = Path("paths.yaml")) -> "MemberPackage":
        """The package a CLI was pointed at, or the one that ships when it was pointed nowhere.

        Every `--root` defaults to None and resolves through here, so the fallback is written once
        rather than as an `or` at each of four call sites that could each drift to a different home.
        """
        return cls(root if root is not None else cls.shipped_root(paths))

    def __init__(self, root: Path, tie: str | None = None) -> None:
        """A mounted package, optionally under a tie rule WE measured rather than one it declares.

        THE OVERRIDE IS AN INFERENCE ABOUT SOMEONE ELSE'S WEIGHTS, and it is an argument rather than a
        default so that it can never be mistaken for something the package said. The published manifest
        carries no `tie` field, so its members would otherwise be routed under OUR first-maximal rule —
        and they reproduce their own recorded predictions markedly better under last-maximal (4 of 4
        members, 3 seeds, mean residual 15-25% lower; knee-evn). Passing it here keeps the claim, its
        evidence and its owner in one place instead of buried in a kernel.

        It applies to every member of the package, which is right for a package whose members share one
        author and one pipeline, and wrong for anything else — so it is opt-in per construction.
        """
        self.root = root
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.members = [
            Member(
                identifier=entry["id"],
                file=root / entry["file"],
                fold=int(entry["fold"]),
                config=entry["config"] if tie is None else {**entry["config"], "tie": tie},
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

        MEASURED 2026-08-17, and the ordering is not merely weak — it is not the right question. Over
        the thirteen checkpoints that share the corpus split, `deep24` scores 0.8352 alone, BELOW the
        pool's floor of 0.8459, and joint selection takes it at every subset size from four up; `crop80`
        scores 0.8345, within 0.0007 of it, and joint selection takes it at none. Solo score does not
        separate the two. Prefer `named` with a subset chosen jointly offline whenever one exists, and
        read this sort as the fallback for when nobody has done that work.

        A member with NO recorded holdout is ranked LAST rather than compared, because a full-data fit
        has none by construction and NaN compares False against every number — under a plain descending
        sort it lands FIRST and `strongest(1)` returns the one member that was never scored. Naming an
        unranked member for a submission stays a decision someone makes, not one a sort makes for them.
        """
        ranked = sorted(self.members, key=lambda member: (not math.isnan(member.holdout), member.holdout), reverse=True)
        unranked = [member.identifier for member in ranked if math.isnan(member.holdout)]
        if unranked and count > len(ranked) - len(unranked):
            logger.warning("members with no recorded holdout, ranked last: %s", ", ".join(unranked))
        return ranked[:count]

    def mounts(self, members: Sequence["Member"] | None = None) -> list[str]:
        """Every Kaggle source a kernel must attach to run THESE members, deduplicated and sorted.

        WHICH members is an argument because a kernel need not run the package: an entry that names a
        two-member subset needs those two encoders and nothing else, so asking about the whole package
        would flag mounts it is right not to declare. Measured over the ten kernels in `comp/kernels`
        on 2026-08-17: three of them run subsets and would read as undeclared against the full package.

        The weights a member was fitted on are not in its checkpoint — only the head is, and the
        backbone is reconstructed from `variant` at load time. So a package's real dependency is the set
        of encoder mounts its members name, and that set is derived from the members rather than typed
        into a kernel's metadata by hand.

        A variant with no mount is refused HERE, as a `WeightsError` rather than the `ValueError`
        `mounted_source` raises on its own. The distinction is not cosmetic: every caller of this method
        is a pre-flight check that already handles `WeightsError` by returning a refusal and carrying on
        to the next kernel, and handles nothing else at all — so an unmounted variant used to escape as
        a traceback that left the REMAINING kernels unchecked. A shippability failure and a
        weights-mismatch failure are the same kind of answer to the same question: these weights cannot
        be trusted to run under this entry.
        """
        chosen = self.members if members is None else members
        adrift = [member for member in chosen if ENCODERS[member.variant].mount is None]
        if adrift:
            unmounted = sorted({member.identifier for member in adrift})
            variants = sorted({member.variant for member in adrift})
            message = (
                f"{unmounted} sit on {variants}, which no Kaggle dataset mirrors, so they can be trained "
                "here and not submitted until their weights are published"
            )
            raise WeightsError(message)
        return sorted({ENCODERS[member.variant].mounted_source for member in chosen})

    def named(self, wanted: Sequence[str]) -> list["Member"]:
        """Exactly these members, in the order asked for, or a refusal naming what is not mounted.

        WHY AN ENTRY NAMES ITS MEMBERS RATHER THAN ASKING FOR THE STRONGEST FEW. `strongest` sorts by
        each member's own solo holdout, which is a MARGINAL ranking, and an ensemble is not marginal:
        measured 2026-08-16 on the 909-study fold-0 holdout over eight of our cached members, the four
        that sort strongest score 0.8588 together, while the best four chosen by their JOINT score
        reach 0.8612 — the sort drops `intensity392_ws` (solo 0.8459) for `extended_ws` (solo 0.8462)
        and loses 0.0024 doing it. A member earns its seat by what it adds, not by what it scores alone.

        Missing names are FATAL rather than skipped. Dropping silently would ship a smaller entry than
        the one whose offline score justified the slot, and nothing in the run's output would say which
        ensemble actually predicted — the same class of failure the fingerprint check exists to stop.
        """
        available = {member.identifier: member for member in self.members}
        absent = [name for name in wanted if name not in available]
        if absent:
            raise WeightsError(f"{self.root} holds {sorted(available)} and was asked for {absent}")
        return [available[name] for name in wanted]

    @classmethod
    def find(cls, mounts: "Mounts | None" = None, name: str = "manifest.json") -> "MemberPackage | None":
        """Locate a mounted package, or return None when none is attached.

        The manifest decides, not the filenames beside it: testing for a naming convention makes the
        search agree with whatever the packager last called its files, which is a second definition of
        what a package is.

        `Mounts` does the walking because the competition's image directories have to be PRUNED, not
        filtered: this read `base.rglob(name)` and skipped the study paths it yielded, which pays the
        entire cost of descending into them and then throws the result away.
        """
        finding = mounts or Mounts()
        if not finding.root.is_dir():
            return None
        holding = finding.every(name)
        return cls(holding[0]) if holding else None

    def load(self, member: Member, backbone: nn.Module, dim: int, device: torch.device) -> StudyModel:
        """Build the member's model, load its weights, and refuse it if the fingerprint disagrees."""
        # The member records BOTH its encoder and the size it read at, and nothing until now checked
        # that the two are compatible: DINOv2 tiles 392 and DINOv3 does not, so a member could name a
        # combination no patch grid can express and only fail inside a kernel.
        ENCODERS[member.variant].require_tiling(member.read_size)
        # THE SHARED REBUILD — including the adapters, which must exist before the weights arrive: a
        # LoRA member's state dict carries adapter keys that only exist once `low_rank` has wrapped the
        # encoder's linears, and loading into the plain architecture fails on unexpected keys at best,
        # at worst returning the pretrained base with no adaptation at all. This path spelled its own
        # subset of the config and so did the ranking tool's; the anatomical prior was the one this
        # half had never heard of.
        model = StudyModel.from_config(member.config, backbone, dim, member.slot_count).to(device)
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
