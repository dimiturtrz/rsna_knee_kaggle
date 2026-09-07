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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray

from core.data.corpus.crop_centre import CROP_MM, DEFAULT_CENTRE, Crop
from core.data.corpus.slot_reader import SPREAD, Band, Reading
from core.data.corpus.slots import CANONICAL_SCHEME, FIRST_MAXIMAL, SCHEMES, Routing
from core.data.corpus.study_cache import StudyCache
from kneemri.models.backbone import SMALL
from kneemri.models.model import StudyDesign

logger = logging.getLogger(__name__)


LEGACY_SLICES = 12
"""What a manifest written before members could differ in depth was fitted on."""

RESIZED_FROM = "resized_from"
"""The size a member was FITTED at, stamped by `resized` so `TrainerRecord` refuses its native array."""

CACHE_SCHEME = "cache_scheme"
"""The scheme naming the cache to READ, when the recorded slot list cannot name it on its own.

`canonical`, `merged` and `merged3` hold the SAME six slots and differ only in which studies they hold,
so `scheme`'s lookup — first scheme whose slot names match — answers `canonical` for all three, and no
checkpoint on disk records a scheme key to break the tie. Every member fitted on a merged cache
therefore resolves to the canonical arrays at serve time: the ninth instance of a value travelling in a
run's options while its consumer reads something else, and the first where the consumer is a LOOKUP
rather than a constant. Harmless for holdout scoring, because our studies lead every merged cache in
the same order — and silently wrong the moment anything wants a study only a merged cache holds.

Set by `on_scheme`, never by the trainer, so a member on disk keeps resolving exactly as it did."""


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

    resized_from: NotRequired[int]  # see RESIZED_FROM; absent on a member served exactly as fitted

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
    fold_recorded: bool = True
    """Whether `fold` was READ from the checkpoint or defaulted — see `held_out_from`, which acts on it.
    Defaults True so the package manifest, which always records a fold, is unaffected."""

    @property
    def reads_pixels(self) -> bool:
        """Whether this checkpoint records a reading policy at all — i.e. whether it is a member.

        The processed directory holds artifacts that were never members: the SSL encoders, whose config
        carries `mask_ratio` and `spectral_weight` instead of a read size, and `vjepa_member_e5.encoder`,
        which carries the member it was cut from. They have no head, cannot be scored and cannot be
        seated, but they sit beside the members and every tool that globbed `*.pt` died on the first one
        with `KeyError: 'img'` before reading a single real member (knee, 2026-08-30).

        `img` is the honest test because every member records the size it was fed; an artifact without
        one has no reading to group by, which is exactly what makes it not a member.
        """
        return "img" in self.config

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

        A ViT interpolates its position embeddings, so what it SPENDS reading at this size is what a
        member fitted at it spends — which is how a resolution is priced before that arm exists (knee-d2fc).
        THE SIZE HAS TO BE THE MEMBER'S, not the caller's tensor: `StudyModel.forward` resamples its input
        to the `img_size` from `window(...).size`, so 448 px pixels under a member still saying 392 measure
        a resize plus an encoder at the OLD resolution — 392, 420 and 448 all read 380 ms (2026-08-17).
        """
        return replace(self, config={**self.config, "train_size": size, RESIZED_FROM: self.read_size})

    def over_pixels(self, size: int) -> "Member":
        """The same member READ OUT OF a cache stored at another size, still fed the size it was fitted at.

        The exact mirror of `resized`, and the two are not interchangeable: `resized` keeps the cache and
        changes what the encoder is fed (pricing a resolution), this keeps what the encoder is fed and
        changes which array the pixels come out of. `StudyModel.forward` resamples its input to
        `window(...).size`, which is `read_size` — so pinning `train_size` to the fitted size is what makes
        336 px on disk arrive at a 224 px encoder as the resize it is, rather than as a different member.

        IT EXISTS BECAUSE A CORPUS HAS A GEOMETRY AND A MEMBER DOES NOT GET TO CHOOSE IT. `on_scheme` moves
        a member onto another corpus's cache and says nothing about size, so a 224 px twin pointed at
        `merged3` resolved to `images_224px_12sl_merged3.npy` — a file that has never existed, because
        merged3 was only ever built at 336. That is the same defect class as a member resolving to the
        DEFAULT arrays: both sides agree they read "the cache" and name different files. It cost the
        teacher pass at 04:56Z on 2026-09-07, and with it the noisy_student arm gated behind it.

        Identity when the sizes already agree, so a caller may always state the corpus's size without
        stamping `RESIZED_FROM` on a member that reads its own pixels — that mark is what stops
        `TrainerRecord.instead_of` handing back a stored holdout, and a native read has earned its record.
        """
        if size == self.image_size:
            return self
        return replace(
            self, config={**self.config, "img": size, "train_size": self.read_size, RESIZED_FROM: self.image_size}
        )

    def on_scheme(self, scheme: str) -> "Member":
        """The same member READ over a DIFFERENT corpus's cache — for TEACHING studies its own cache
        does not hold, never for a number compared against its own.

        A teacher must read every study the student will be fitted on, and a student fitted on a merged
        corpus is fitted on studies no canonical cache holds. Without this the join in
        `TeacherFile.aligned` raises on the foreign rows — correctly, since the alternative is a student
        taught another study's reading — so the teacher has to be pointed at the STUDENT's cache.

        SCHEME ONLY, and the depth is not its business: a merged cache also differs from the member's
        own in slices and band, and `SharedRead.regrouped` already re-points those for the kernel and
        for `holdout_backfill`. A second spelling of the regroup here would be two definitions of what a
        member reads. Compose them — regroup for the geometry, this for which studies exist.
        """
        return replace(self, config={**self.config, CACHE_SCHEME: scheme})

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
    def held_out_from(cls, fold: int, checkpoints: list[Path]) -> list["Member"]:
        """The members that did NOT train on the fold being scored, with the rest named and dropped.

        A member records the fold it held out. One fitted under a different fold — `--fold -1` above all,
        which holds out nothing — trained on the studies a holdout tool is about to score it on, so its
        number is in-sample and reads high: `base_dense_ff` came back 0.9035 on 2026-08-30 against every
        honest member's 0.83-0.86. `member_curve` seats in SCORE ORDER, so a leaked member is seated
        FIRST and the whole ordering is junk, silently, in a log indistinguishable from a clean sweep.
        `record_mix` has always refused these (`no holdout record, skipped`) and `member_curve` averaged
        them in; the asymmetry between two tools answering one question is the defect (knee-vq46).

        A member that records NO fold is refused too — `from_checkpoint` defaults those to 0, which is
        the same leak reopened by a default. Thirteen checkpoints carry no `options.fold`: nine named
        `*_fold0` where the default is right, and `member_fold1` … `member_fold4` where it is wrong. So
        the refusal costs nine honest members out of 162, and it is worth the loss — reading a fold off
        the FILENAME would be guessing dressed as a rule, and `MemberReadings` already refuses a cache
        whose study order disagrees rather than averaging over it.
        """
        named = [cls.from_checkpoint(path) for path in checkpoints]
        for member in named:
            if not member.fold_recorded:
                logger.info("%s: records no fold, cannot be shown held out — skipped", member.file.name)
            elif member.fold != fold:
                logger.info("%s: fitted at fold %d, trained on these studies — skipped", member.file.name, member.fold)
        return [member for member in named if member.fold_recorded and member.fold == fold]

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
            fold_recorded="fold" in saved.get("options", {}),
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

        A STATED scheme wins over the lookup, because the lookup cannot always answer: three schemes
        share the canonical slot list and differ only in which studies they hold, so the tie goes to
        whichever `SCHEMES` names first. See `CACHE_SCHEME` for what that costs and who sets it.
        """
        stated = self.config.get(CACHE_SCHEME)
        if stated is not None:
            return str(stated)
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
