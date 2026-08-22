import json
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast, override

import numpy as np
import pytest
import torch
from torch import Tensor

from core.data.corpus.crop_centre import CROP_MM, DEFAULT_CENTRE
from core.data.corpus.slot_reader import Band, Reading
from core.data.corpus.slots import SCHEMES, Routing
from core.mounts import Mounts
from core.paths import COMPETITION
from kneemri.inference.members import (
    LEGACY_SLICES,
    SHIPPED_PACKAGE,
    Member,
    MemberConfig,
    MemberPackage,
    WeightsError,
)
from kneemri.models.backbone import SMALL
from kneemri.models.model import StudyDesign, StudyModel

from ..runs.member_record import OPTIONS
from ..runs.member_record import _config as _recorded_config  # the WRITER's own output, never a copy

SEED = 2026
DIM = 8
PATCHES = 16
SLOTS = ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS", "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"]
IMAGE = 14
GROUP = 3


@dataclass
class _Output:
    last_hidden_state: Tensor


class _FakeBackbone(torch.nn.Module):
    """Deterministic stand-in encoder — the fingerprint only needs the forward path to be stable."""

    @override
    def forward(self, *, pixel_values: Tensor) -> _Output:
        n = pixel_values.shape[0]
        column = pixel_values.mean(dim=(1, 2, 3)).reshape(n, 1, 1)
        base = torch.arange(PATCHES + 1, dtype=torch.float32).reshape(1, -1, 1)
        return _Output(last_hidden_state=(base + column).expand(n, PATCHES + 1, DIM).contiguous())


def _config() -> MemberConfig:
    return {
        "img": IMAGE,
        "slices": 12,
        "group": GROUP,
        "crop_mm": 130.0,
        "band": [0.2, 0.8],
        "slots": SLOTS,
        "variant": "small",
        "pool": "cls_mean",
    }


def _package(tmp_path: Path, *, corrupt: bool = False) -> tuple[MemberPackage, torch.device]:
    """A one-member package built from a model we just made, so the fingerprint is self-consistent."""
    device = torch.device("cpu")
    torch.manual_seed(SEED)
    model = StudyModel(_FakeBackbone(), DIM, len(SLOTS))
    stored = model.fingerprint(IMAGE, GROUP, device)
    if corrupt:
        stored = stored + 1.0
    torch.save(model.state_dict(), tmp_path / "m_test.pt")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format": 2,
                "members": [
                    {
                        "id": "test",
                        "file": "m_test.pt",
                        "fold": 0,
                        "config": _config(),
                        "fingerprint": stored.tolist(),
                        "holdout": 0.84,
                        "annot": 0.87,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return MemberPackage(tmp_path), device


def test_check(tmp_path: Path):
    """A faithful model passes with slack; a stored fingerprint that disagrees is fatal, not a warning."""
    package, device = _package(tmp_path)
    member = package.members[0]
    torch.manual_seed(SEED)
    model = StudyModel(_FakeBackbone(), DIM, len(SLOTS))
    assert package.check(model, member, device) == pytest.approx(0.0, abs=1e-6)

    corrupt_package, _ = _package(tmp_path, corrupt=True)
    with pytest.raises(WeightsError, match="fingerprint differs"):
        corrupt_package.check(model, corrupt_package.members[0], device)

    # A DIFFERENT SHAPE IS A DIFFERENT ARCHITECTURE, and it must not reach the subtraction: numpy would
    # broadcast a (2, 1) stored answer against a (2, 12) one and report a difference small enough to pass.
    reshaped = replace(member, fingerprint=member.fingerprint[:, :1])
    with pytest.raises(WeightsError, match="not the one these weights were fitted to"):
        package.check(model, reshaped, device)


def test_load(tmp_path: Path):
    """Loading returns a model whose weights are the member's and whose fingerprint agrees."""
    package, device = _package(tmp_path)
    model = package.load(package.members[0], _FakeBackbone(), DIM, device)
    assert isinstance(model, StudyModel)


def test_mounts(tmp_path: Path):
    """A package states the encoder sources a kernel must attach, derived from its members.

    Derived rather than declared: the backbone is NOT in a checkpoint — only the head is — so a kernel
    that was never told to mount `small` fails deep inside a load with the slot already spent.
    """
    package, _ = _package(tmp_path)
    assert package.mounts() == ["metaresearch/dinov2/PyTorch/small/1"]
    assert package.mounts(package.named(["test"])) == package.mounts(), "a subset states only what it needs"
    assert package.mounts([]) == [], "no members need no encoder"


def test_find(tmp_path: Path):
    """The manifest decides where the package is; absence is a path, not an error."""
    assert MemberPackage.find(Mounts(root=tmp_path / "nothing-here")) is None
    nested = tmp_path / "input" / "some-dataset"
    nested.mkdir(parents=True)
    _package(nested)
    found = MemberPackage.find(Mounts(root=tmp_path / "input"))
    assert found is not None
    assert found.root == nested


def test_member_package_refuses_a_manifest_with_absent_files(tmp_path: Path):
    """A manifest listing files that are not there is a broken package, caught before any prediction."""
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "members": [
                    {
                        "id": "gone",
                        "file": "m_gone.pt",
                        "fold": 0,
                        "config": _config(),
                        "fingerprint": [[0.0]],
                        "holdout": 0.0,
                        "annot": 0.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(WeightsError, match="absent"):
        MemberPackage(tmp_path)


def _member() -> Member:
    return Member(
        identifier="x",
        file=Path("m_x.pt"),
        fold=0,
        config=_config(),
        fingerprint=np.zeros((2, 12), np.float32),
        holdout=0.8,
        annot=0.8,
    )


def test_image_size():
    assert _member().image_size == IMAGE


def test_resized():
    """A priced resolution has to reach the WINDOW, which is the size the model resamples its input to.

    Moving only the tensor times a bilinear resize followed by the encoder at the old resolution —
    measured flat at 380 ms for 392, 420 and 448 alike. The original member is untouched, so nothing
    that holds one is quietly re-priced.
    """
    original = _member()
    bigger = original.resized(IMAGE * 2)

    assert bigger.read_size == IMAGE * 2, "what the tensor is allocated at"
    assert bigger.window(bigger.slices, overlap=False).size == IMAGE * 2, "and what the encoder is actually run at"
    assert original.read_size == IMAGE, "the member handed in is not mutated"


def test_lora_rank():
    """A member says whether its encoder was adapted, and every older manifest correctly says no.

    The loader has to know BEFORE it builds the model: a LoRA member's state dict carries adapter keys
    that only exist once `low_rank` has wrapped the encoder's linears, so a default of zero for old
    manifests is not a convenience, it is the correct answer for members that carry no adapters.
    """
    assert _member().lora_rank == 0, "a manifest written before adaptation existed has no adapters"
    adapted = replace(_member(), config=_config() | {"lora_rank": 8})
    assert adapted.lora_rank == 8


def test_read_size():
    """Absent means "the cache's own size", which is what every member written before this key means.

    The equivalence classes are the two a manifest can be in: no `train_size` (every published member
    and both whole-stack members on disk), and one that differs from `img`. Getting the first wrong
    would refuse members that are perfectly valid; getting the second wrong would rebuild a member at
    a resolution it never saw and mispredict rather than fail.
    """
    assert _member().read_size == IMAGE, "no train_size means the model read the cache as decoded"

    resampled = replace(_member(), config={**_config(), "train_size": 392})
    assert resampled.read_size == 392
    assert resampled.image_size == IMAGE, "the cache it decodes from is unchanged"


def test_slot_count():
    assert _member().slot_count == len(SLOTS)


def test_group():
    assert _member().group == GROUP


def test_pool():
    assert _member().pool == "cls_mean"


def test_pixel_group():
    """Members needing identical pixels share one decode, so the key must ignore everything else."""
    same = _member().pixel_group
    other_config = _config() | {"variant": "base"}
    other = Member(
        identifier="y",
        file=Path("m_y.pt"),
        fold=1,
        config=other_config,
        fingerprint=np.zeros((2, 12), np.float32),
        holdout=0.1,
        annot=0.1,
    )
    assert other.pixel_group == same, "a difference that does not change the pixels must not split the decode"
    resized = Member(
        identifier="z",
        file=Path("m_z.pt"),
        fold=2,
        config=_config() | {"img": 224},
        fingerprint=np.zeros((2, 12), np.float32),
        holdout=0.1,
        annot=0.1,
    )
    assert resized.pixel_group != same, "a different resolution is a different decode"

    without_depth = {key: value for key, value in _config().items() if key != "slices"}
    legacy = Member(
        identifier="w",
        file=Path("m_w.pt"),
        fold=3,
        config=cast(MemberConfig, without_depth),
        fingerprint=np.zeros((2, 12), np.float32),
        holdout=0.1,
        annot=0.1,
    )
    stated = Member(
        identifier="v",
        file=Path("m_v.pt"),
        fold=4,
        config=_config() | {"slices": LEGACY_SLICES},
        fingerprint=np.zeros((2, 12), np.float32),
        holdout=0.1,
        annot=0.1,
    )
    # A manifest predating variable depth is SERVED at twelve, so it must be GROUPED at twelve. The key
    # indexed the config raw and raised KeyError here instead — one fact answered two ways, and this is
    # the answer that decides which decode a member shares.
    assert legacy.pixel_group == stated.pixel_group


def test_state_dict_of(tmp_path: Path):
    """A published checkpoint wraps its tensors under `model` beside training metadata; unwrap it."""
    package, device = _package(tmp_path)
    member = package.members[0]
    bare = package.state_dict_of(member, device)
    assert "head.out.weight" in bare

    wrapped = tmp_path / "m_wrapped.pt"
    torch.save({"model": bare, "epoch": 60, "holdout": 0.84, "annot": 0.87}, wrapped)
    assert package.state_dict_of(replace(member, file=wrapped), device).keys() == bare.keys()


def test_strongest(tmp_path: Path):
    """Best holdout first, and never more than asked for.

    The members are varied by replacing the loaded one rather than by writing three checkpoints: the
    ordering is a property of the recorded scores, and saving three identical models to prove it would
    only test `torch.save`.
    """
    package, _ = _package(tmp_path)
    package.members = [
        replace(package.members[0], identifier=str(score), holdout=score) for score in (0.81, 0.87, 0.84)
    ]

    assert [member.holdout for member in package.strongest(2)] == [0.87, 0.84]
    assert len(package.strongest(9)) == 3


def test_named(tmp_path: Path):
    """The named members in the order asked for, and a REFUSAL when one is not mounted.

    Skipping an absent name would ship a smaller ensemble than the one whose offline score justified
    the slot, with nothing in the run's output naming which ensemble actually predicted.
    """
    package, _ = _package(tmp_path)
    package.members = [
        replace(package.members[0], identifier=str(score), holdout=score) for score in (0.81, 0.87, 0.84)
    ]

    assert [member.identifier for member in package.named(["0.84", "0.81"])] == ["0.84", "0.81"]
    with pytest.raises(WeightsError, match="0.99"):
        package.named(["0.84", "0.99"])


def test_the_member_with_no_holdout_is_ranked_last_rather_than_compared(tmp_path: Path):
    """A full-data fit records NaN by construction, and NaN sorts FIRST under a plain descending sort.

    `strongest(1)` is what the solo kernel submits, so this decides which weights a slot runs: the
    unranked member compares False against every number, keeps its place, and `reverse=True` puts it at
    the front. Ranking it last keeps naming it a decision someone makes rather than one a sort makes.
    """
    package, _ = _package(tmp_path)
    package.members = [
        replace(package.members[0], identifier=str(score), holdout=score) for score in (float("nan"), 0.87, 0.84)
    ]

    assert [member.holdout for member in package.strongest(2)] == [0.87, 0.84]
    assert math.isnan(package.strongest(3)[-1].holdout), "unranked, so last — never dropped silently"


def test_variant(tmp_path: Path):
    """A manifest written before variants existed still answers, and answers `small`.

    Every member published so far predates the field, and the kernel indexes its DINOv2 mounts by this
    value — so a missing key has to mean the encoder those members were actually fitted on, not a
    KeyError at submission time.
    """
    package, _ = _package(tmp_path)
    assert package.members[0].variant == SMALL

    package.members = [replace(package.members[0], config={**package.members[0].config, "variant": "base"})]
    assert package.members[0].variant == "base"


def test_slices(tmp_path: Path):
    """A member states the stack depth it was fitted on, and old manifests fall back to twelve.

    The key was in MemberConfig from the start with no accessor to read it - the same gap that left
    `variant` unreadable while the kernel already called it - so inference decoded a hardcoded twelve
    whatever the member had actually seen.
    """
    package, _ = _package(tmp_path)
    member = package.members[0]

    member.config["slices"] = 20
    assert member.slices == 20

    # Re-read a manifest with the key STRIPPED rather than deleting it from the typed config: a legacy
    # manifest on disk is the real case, and `slices` is a required key the type system refuses to drop.
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    del manifest["members"][0]["config"]["slices"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    legacy = MemberPackage(tmp_path).members[0]
    assert legacy.slices == LEGACY_SLICES, "a manifest predating variable depth means the twelve it was fitted on"


def test_slice_pooling():
    """A member's slice aggregation is read from its config, defaulting to the published behaviour.

    It is a different function on different pixels, not a preference: "max" encodes each slice alone
    and takes the per-dimension maximum, "channels" stacks three as RGB. This was recorded by every
    checkpoint and read by nobody, which made every max-pooling member unsubmittable — the fingerprint
    check refused them, correctly, for being a different architecture than the one stored.
    """
    assert _member().slice_pooling == "channels", "published members predate the option and were fitted so"

    maxed = replace(_member(), config={**_config(), "slice_pooling": "max"})
    assert maxed.slice_pooling == "max"


def test_whole_stack():
    """Absent means windowed, which is what every member fitted before the flag existed was.

    The two published-member classes are the equivalence classes that matter: the twenty CC0 members
    carry no such key and must never be served a whole stack, and ours carry it True and must never be
    served three slices at a time.
    """
    assert not _member().whole_stack, "a manifest without the key describes a windowed member"
    assert replace(_member(), config={**_config(), "whole_stack": True}).whole_stack


def test_precision():
    """Absent means float16, which is the arithmetic every member predating the flag actually used.

    The three equivalence classes are the three states a manifest can be in, and they are not
    interchangeable: absent (the twenty published members and everything trained before --bf16),
    explicitly False (a run that had the flag and declined it), and True. Defaulting the first two to
    anything but float16 would serve those members arithmetic they were never fitted under — which is
    precisely the defect this property was added to fix, in the other direction: `predict_member` ran a
    dtype-less autocast, so every bf16 member was served in fp16.
    """
    assert _member().precision is torch.float16, "no bf16 key means the run used autocast's cuda default"
    assert replace(_member(), config={**_config(), "bf16": False}).precision is torch.float16
    assert replace(_member(), config={**_config(), "bf16": True}).precision is torch.bfloat16


def test_witness():
    """Absent means unverified, not verified — the check must say so rather than pass quietly."""
    assert _member().witness == {}
    carried = {"studies": ["a"], "predictions": [[0.1] * 12]}
    assert replace(_member(), config={**_config(), "witness": carried}).witness == carried


def _saved(tmp_path: Path, name: str, config: dict[str, object]) -> Path:
    path = tmp_path / f"{name}.pt"
    torch.save({"model": {}, "config": config, "holdout": 0.84}, path)
    return path


def test_from_checkpoint(tmp_path: Path):
    """A member read straight from a training checkpoint is the SAME type inference uses.

    Analysis kept re-deriving these fields from raw checkpoints and getting one wrong each time — the
    read size, the slot count, whether the whole stack is read. One description, one place.
    """
    path = _saved(
        tmp_path,
        "arm",
        {
            "img": 448,
            "train_size": 392,
            "slices": 12,
            "whole_stack": True,
            "variant": "small",
            "slots": SLOTS,
            "group": GROUP,
            "pool": "cls_mean",
            "crop_mm": 130.0,
            "band": [0.2, 0.8],
        },
    )
    member = Member.from_checkpoint(path)
    assert (member.image_size, member.read_size, member.whole_stack) == (448, 392, True)
    assert member.holdout == pytest.approx(0.84)
    assert member.identifier == "arm"


def test_window(tmp_path: Path):
    """A whole-stack member reads one centred full-depth window; a windowed one reads its groups.

    Overlap is a choice about WINDOWED members. A whole-stack member has nothing to overlap: its slot
    embedding is the maximum over the whole stack, and reading it three slices at a time measures a
    different function — which is exactly how it reached the leaderboard (knee-y8y).
    """
    whole = replace(_member(), config={**_config(), "whole_stack": True})
    window = whole.window(16, overlap=True)
    assert window.starts == [2] and window.width == 12, "the middle twelve of sixteen, in one pass"

    windowed = _member()
    assert len(windowed.window(12, overlap=True).starts) == 12 - GROUP + 1
    assert windowed.window(12, overlap=False).starts == [0, 3, 6, 9]
    assert windowed.window(12, overlap=True).width == GROUP


def test_scheme():
    """A member's slot list names its scheme, which is how its cache is found.

    Two tools had grown their own copy of this lookup — the member curve and the witness check — which
    is exactly how a description drifts from the thing it describes.
    """
    assert _member().scheme == "canonical"
    extended = replace(_member(), config={**_config(), "slots": [*SLOTS, "COR_FLUID_NOFS", "AX_T1", "AX_FLUID_NOFS"]})
    assert extended.scheme == "extended"
    # The literals above are deliberate: a test comparing against the constant the subject imports only
    # proves the copy agrees with itself. THIS is the property they protect — the fallback a member gets
    # when nothing matches must still be a scheme that RESOLVES, because the name is what finds the cache.
    assert _member().scheme in SCHEMES


def test_layer():
    """Absent means the LAST block, which is what every member fitted before the flag actually read.

    Rebuilding a member on a different block than it was fitted on is a different function that loads
    cleanly and predicts from features it never saw - the ninth-instance shape. So this is read back by
    both rebuilders rather than defaulted at either.
    """
    assert _member().layer == -1, "a manifest without the key describes a member that read the last block"
    assert replace(_member(), config={**_config(), "encoder_layer": 6}).layer == 6


def test_slot_pooling():
    """Read back from the member's own config, defaulting to attention for everything fitted before.

    Wired to its consumer before the ablation arm was queued rather than after it returned a number: a
    head fitted with uniform pooling and rebuilt with per-diagnosis attention loads without complaint,
    which is the shape that has cost this project nine defects.
    """
    member = _member()

    assert member.slot_pooling == "attend", "every member on disk predates the flag and was attended"
    assert replace(member, config={**member.config, "slot_pooling": "mean"}).slot_pooling == "mean"


def test_band():
    """The rule, from the list JSON can hold — and it is a grouping key, so it must be hashable."""
    wider = replace(_member(), config={**_config(), "band": [0.1, 0.9]})
    legacy = replace(_member(), config={key: value for key, value in _config().items() if key != "band"})

    assert wider.band == Band(0.1, 0.9)
    assert hash(wider.band), "the read groups by it, which a list cannot do"
    assert legacy.band == Band(), "a manifest without the field is the historic reading, not a crash"


def test_crop():
    """The physical extent a member was fitted on, defaulting to the constant for older manifests.

    Read by NOTHING until 2026-08-14: the decode built its reader from the module constant, so a member
    fitted on a tighter crop was served under 130 mm and every fingerprint still passed, because each is
    checked against its own reading.
    """
    tighter = replace(_member(), config={**_config(), "crop_mm": 80.0})
    assert tighter.crop.mm == 80.0

    older = replace(_member(), config={key: value for key, value in _config().items() if key != "crop_mm"})
    assert older.crop.mm == CROP_MM, "a manifest written before the crop could vary means the default"

    located = replace(_member(), config={**_config(), "centre": "tissue"})
    assert located.crop.centre == "tissue"
    assert _member().crop.centre == DEFAULT_CENTRE, "a manifest that does not say means the historic centre"
    assert located.pixel_group != _member().pixel_group, "different pixels cannot share one decode"


def test_reading():
    """Both geometry axes together, because the pair is what names the cache a member is served from."""
    member = replace(_member(), config={**_config(), "crop_mm": 80.0, "band": [0.1, 0.9]})

    assert member.reading == Reading(crop=member.crop, band=member.band)
    assert member.reading.suffix() == "_80mm_0.1-0.9band", "either axis dropped here is a silent mismatch"
    assert _member().reading == Reading(), "a historic manifest reads the arrays it was fitted on"


def test_cache(tmp_path: Path):
    """The cache a member is served from must be named by the WHOLE reading it records.

    THE DEFECT THIS PINS: three call sites built this by hand from size, depth and router, so a member
    recording any other crop or band resolved to the default arrays — which exist, load, and are the
    wrong pixels. Nothing raises on that path; the fingerprint passes, because it is checked against the
    member's own reading rather than against what it was fed.
    """
    tighter = replace(_member(), config={**_config(), "crop_mm": 80.0, "band": [0.1, 0.9]})

    assert _member().cache(tmp_path).key == f"{IMAGE}px_12sl", "the historic reading keeps the historic name"
    assert tighter.cache(tmp_path).key == f"{IMAGE}px_12sl_80mm_0.1-0.9band"
    assert tighter.cache(tmp_path).images_path != _member().cache(tmp_path).images_path


def test_cache_studies(tmp_path: Path):
    """A member fitted on a TRUNCATED cache resolves that cache, not the full one it is a prefix of.

    The two differ in one axis length and in their index, so a smoke member re-read against the full
    arrays is scored on another study's pixels per position — the crop-and-band defect on the one axis
    that changes WHICH study a position names.
    """
    smoked = replace(_member(), config={**_config(), "cache_studies": 60})

    assert _member().cache_studies == 0, "a manifest written before truncation existed read the full corpus"
    assert smoked.cache(tmp_path).key == f"{IMAGE}px_12sl_first60", "it resolves the cache it was fitted on"


def test_from_checkpoint_reads_a_published_slot_list(tmp_path: Path):
    """A published checkpoint records its slots at the TOP LEVEL, not inside config.

    Ours put them in `config`; theirs do not have the key at all, so `scheme` and `slot_count` raised
    KeyError on every published checkpoint. It went unnoticed because published members normally arrive
    through a manifest that supplies the list — the first tool to open one of their .pt files directly
    was `oof_oracle`, whose whole purpose is to compare their pipeline against ours.
    """
    slots = list(SLOTS)
    path = tmp_path / "m_published.pt"
    torch.save({"config": {"img": 336, "slices": 12}, "slots": slots, "holdout": 0.84}, path)

    member = Member.from_checkpoint(path)

    assert member.config["slots"] == slots, "taken from where they actually live"
    assert member.slot_count == len(slots)
    assert member.scheme == "canonical", "and the scheme resolves, which is what finds its cache"


def test_tie():
    """Which router a member was fitted under, and that a different one is different PIXELS."""
    theirs = replace(_member(), config={**_config(), "tie": "last"})

    assert theirs.tie == "last"
    assert _member().tie == "first", "ours, and what every cache on disk holds"
    assert theirs.pixel_group != _member().pixel_group, "a different router cannot share a decode"


def test_routing():
    """A member's routing is the pair its pixels depend on: which slots, and who wins a tie."""
    theirs = replace(_member(), config={**_config(), "tie": "last"})

    assert _member().routing == Routing(scheme="canonical", tie="first")
    assert theirs.routing.tie == "last", "recorded per member, because ours and theirs differ"


def test_member_package_takes_a_measured_tie_override(tmp_path: Path):
    """A package can be read under a tie rule WE measured, since theirs declares none.

    An argument rather than a default: it is an inference about someone else's weights, and burying it
    in a kernel would hide the claim from the manifest it contradicts.
    """
    _package(tmp_path)

    plain = MemberPackage(tmp_path)
    assert plain.members[0].tie == "first", "what a manifest that says nothing about it means"

    overridden = MemberPackage(tmp_path, tie="last")
    assert overridden.members[0].tie == "last"
    assert overridden.members[0].pixel_group != plain.members[0].pixel_group, "different pixels"


CONSUMED: dict[str, tuple[object, Callable[[Member], object], object]] = {
    "img": (448, lambda member: member.image_size, 448),
    "train_size": (392, lambda member: member.read_size, 392),
    "slices": (20, lambda member: member.slices, 20),
    "cache_studies": (60, lambda member: member.cache_studies, 60),
    "group": (5, lambda member: member.group, 5),
    "crop_mm": (80.0, lambda member: member.crop.mm, 80.0),
    "centre": ("tissue", lambda member: member.crop.centre, "tissue"),
    "tie": ("last", lambda member: member.tie, "last"),
    "band": ([0.1, 0.9], lambda member: member.band, Band(low=0.1, high=0.9)),
    "variant": ("base", lambda member: member.variant, "base"),
    "pool": ("cls_mean_focal", lambda member: member.pool, "cls_mean_focal"),
    "slice_pooling": ("max", lambda member: member.slice_pooling, "max"),
    "slot_pooling": ("mean", lambda member: member.slot_pooling, "mean"),
    "anatomical_prior": (True, lambda member: StudyDesign.from_config(member.config).anatomical_prior, True),
    "slice_regions": (4, lambda member: StudyDesign.from_config(member.config).slice_regions, 4),
    "through_plane": (True, lambda member: StudyDesign.from_config(member.config).through_plane, True),
    "encoder_layer": (-4, lambda member: member.layer, -4),
    "lora_rank": (8, lambda member: member.lora_rank, 8),
    "whole_stack": (True, lambda member: member.whole_stack, True),
    "bf16": (True, lambda member: member.precision, torch.bfloat16),
    "witness": ({"studies": ["A"]}, lambda member: member.witness, {"studies": ["A"]}),
    "slots": (["SAG_T1", "COR_T1"], lambda member: member.slot_count, 2),
}
"""Every recorded field a SERVE-side reader must take from the member, as (probe, reader, expected).

The probe is a second legal value, so a reader that quietly used its own default answers the first one.
"""

PROVENANCE_ONLY = {
    "ema",
    "rank_weight",
    "pair_memory",
    "report_weight",
    "unfreeze_last",
    "head_lr",
    "backbone_lr",
    "layer_decay",
}
"""Recorded so the ledger can say which arm produced a member, and read by NOTHING at serve time — a
training rule leaves no trace in the forward pass. Listed rather than omitted, so a field that gains a
consumer moves between the two sets deliberately."""


def test_every_recorded_field_is_either_consumed_or_declared_unread():
    """The mirror of the writer's one-hot walk, on the side that has cost this project nine defects.

    All nine are the same shape: a value written faithfully into the config and a consumer reading a
    module default instead — whole-stack reading windowed to three slices, the decode resolution and the
    slot list each taken from a constant, every bf16 member served in fp16. Each was found by a witness,
    a refused fingerprint or a lost submission, which is to say AFTER a run.

    The writer is already guarded: `test_payload_takes_each_config_field_from_the_option_of_its_own_name`
    moves one option and demands exactly one stamped field follow. Nothing guarded the other end, so a
    key could be recorded correctly and read by nobody — which is precisely what happened, nine times.

    Two claims, and the first is the one that does not rot: every key the writer stamps is classified,
    so a NEW recorded field cannot arrive with no consumer and no stated reason for having none.
    """
    recorded = set(_recorded_config(OPTIONS))

    assert recorded == set(CONSUMED) | PROVENANCE_ONLY, "a recorded field is consumed at serve time or declared unread"

    for name, (probe, read, expected) in CONSUMED.items():
        member = replace(_member(), config={**_config(), name: probe})

        assert read(member) == expected, f"config[{name!r}] is recorded and its serve-side reader ignores it"


def _shipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A paths.yaml naming a data root under tmp_path, with the env override that outranks it removed."""
    monkeypatch.delenv("KNEEMRI_DATA", raising=False)
    paths = tmp_path / "paths.yaml"
    paths.write_text(f"data: {(tmp_path / 'data').as_posix()}\n", encoding="utf-8")
    return paths


def test_shipped_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The one home is under the PROCESSED root, not the repo — the mirror it replaced aged two members behind."""
    root = MemberPackage.shipped_root(_shipped(tmp_path, monkeypatch))

    assert root.name == SHIPPED_PACKAGE
    assert root.parent == tmp_path / "data" / "processed" / COMPETITION


def test_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An explicit root wins; pointing a CLI nowhere resolves the package that ships, never the caller's cwd."""
    paths = _shipped(tmp_path, monkeypatch)
    (tmp_path / "named").mkdir()
    named, _ = _package(tmp_path / "named")
    shipped = MemberPackage.shipped_root(paths)
    shipped.mkdir(parents=True, exist_ok=True)
    (shipped / "manifest.json").write_text(json.dumps({"format": 2, "members": []}), encoding="utf-8")

    assert MemberPackage.at(named.root, paths).root == named.root
    assert MemberPackage.at(None, paths).root == shipped
