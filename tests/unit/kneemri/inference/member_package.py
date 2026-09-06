"""A shelf of weights: what it holds, what it refuses, and which mounts its members need."""

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from core.mounts import Mounts
from core.paths import COMPETITION
from kneemri.inference.member_package import SECOND_SHELF, SHIPPED_PACKAGE, MemberPackage, WeightsError
from kneemri.models.model import StudyModel

from .members import DIM, SEED, SLOTS, _config, _FakeBackbone, _package


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


def test_mounts_refuses_an_unmounted_variant_as_a_weights_error_not_a_traceback(tmp_path: Path):
    """The refusal has to arrive as the error every caller already handles, or it checks nothing.

    Both callers are pre-flight sweeps that answer per kernel and catch only `WeightsError`. While the
    raise was `mounted_source`'s bare `ValueError`, an unmounted member did not produce a REFUSED line —
    it escaped the sweep and left every REMAINING kernel unchecked.
    """
    package, _ = _package(tmp_path)
    unmounted = replace(package.members[0], config={**package.members[0].config, "variant": "cnxbase"})
    with pytest.raises(WeightsError, match="no Kaggle dataset mirrors"):
        package.mounts([unmounted])


def test_find(tmp_path: Path):
    """The manifest decides where the package is; absence is a path, not an error."""
    assert MemberPackage.find(Mounts(root=tmp_path / "nothing-here")) is None
    nested = tmp_path / "input" / "some-dataset"
    nested.mkdir(parents=True)
    _package(nested)
    found = MemberPackage.find(Mounts(root=tmp_path / "input"))
    assert found is not None
    assert found.root == nested


def test_spanning(tmp_path: Path):
    """Two mounted roots become ONE package holding both their members, and absence stays a path.

    This is what lets a member trained today reach the board: the shipped package is ~10 GB, so the new
    checkpoint ships in a dataset of its own, mounted beside it. `find` took the first manifest and made
    re-pushing all ten gigabytes the price of one member.
    """
    assert MemberPackage.spanning(Mounts(root=tmp_path / "nothing-here")) is None
    mounted = tmp_path / "input"
    first = mounted / "kneemri-members"
    second = mounted / "kneemri-members-2"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    _package(first, identifier="shipped")
    _package(second, identifier="fresh")

    spanned = MemberPackage.spanning(Mounts(root=mounted))
    assert spanned is not None
    assert sorted(member.identifier for member in spanned.members) == ["fresh", "shipped"]
    assert sorted(spanned.roots) == sorted([first, second])
    for member in spanned.members:
        assert member.file.is_file(), "a member's file resolves against ITS OWN root, not the first one"


def test_absorb(tmp_path: Path):
    """Members merge; an identifier both roots hold is fatal rather than decided by mount order.

    It takes the MEMBERS and the root rather than the other package, so what is merged is a shelf of
    weights and not another package's behaviour.
    """
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    shipped, _ = _package(first, identifier="shipped")
    fresh, _ = _package(second, identifier="fresh")
    shipped.absorb(fresh.members, second)
    assert sorted(member.identifier for member in shipped.members) == ["fresh", "shipped"]

    stale = tmp_path / "three"
    stale.mkdir()
    duplicate, _ = _package(stale, identifier="fresh")
    with pytest.raises(WeightsError, match="both hold"):
        shipped.absorb(duplicate.members, stale)


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


def _shipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A paths.yaml naming a data root under tmp_path, with the env override that outranks it removed."""
    monkeypatch.delenv("KNEEMRI_DATA", raising=False)
    paths = tmp_path / "paths.yaml"
    paths.write_text(f"data: {(tmp_path / 'data').as_posix()}\n", encoding="utf-8")
    return paths


def test_shipped_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The one home is under the MODELS root, not the repo — the mirror it replaced aged two members behind."""
    root = MemberPackage.shipped_root(_shipped(tmp_path, monkeypatch))

    assert root.name == SHIPPED_PACKAGE
    assert root.parent == tmp_path / "data" / "models" / COMPETITION


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


def test_shipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The local package as a KERNEL mounts it — the shipped shelf plus the shelf beside it.

    A pre-flight built on `at` checks a package no kernel runs: it refused every entry naming a member
    on the second shelf while the entry was correct. An explicit `--root` is still exactly one shelf.
    """
    paths = _shipped(tmp_path, monkeypatch)
    root = MemberPackage.shipped_root(paths)
    root.mkdir(parents=True)
    _package(root, identifier="shipped")
    beside = root.parent / SECOND_SHELF
    beside.mkdir()
    _package(beside, identifier="fresh")

    spanned = MemberPackage.shipped(None, paths)
    assert sorted(member.identifier for member in spanned.members) == ["fresh", "shipped"], "both shelves"
    assert spanned.roots == [root, beside], "in mount order"

    named = MemberPackage.shipped(root, paths)
    assert [member.identifier for member in named.members] == ["shipped"], "--root names a package, never grows one"
