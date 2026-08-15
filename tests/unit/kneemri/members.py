import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import override

import numpy as np
import pytest
import torch
from torch import Tensor

from kneemri.backbone import SMALL
from kneemri.crop_centre import CROP_MM, DEFAULT_CENTRE
from kneemri.members import LEGACY_SLICES, Member, MemberConfig, MemberPackage, WeightsError
from kneemri.model import StudyModel
from kneemri.mounts import Mounts
from kneemri.slots import SCHEMES, Routing

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


def test_load(tmp_path: Path):
    """Loading returns a model whose weights are the member's and whose fingerprint agrees."""
    package, device = _package(tmp_path)
    model = package.load(package.members[0], _FakeBackbone(), DIM, device)
    assert isinstance(model, StudyModel)


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
    """A pair, from the list JSON can hold — and it is a grouping key, so it must be hashable."""
    wider = replace(_member(), config={**_config(), "band": [0.1, 0.9]})

    assert wider.band == (0.1, 0.9)
    assert hash(wider.band), "the read groups by it, which a list cannot do"


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
