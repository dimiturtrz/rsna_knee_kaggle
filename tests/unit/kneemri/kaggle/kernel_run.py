"""What a kernel does once it can import us — without a Kaggle mount."""

import json
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch

from core.mounts import Mounts
from kneemri.inference.ensemble import Serving
from kneemri.inference.test_time import SpatialViews
from kneemri.inference.vote import LogitAverage
from kneemri.kaggle import kernel_run
from kneemri.kaggle.kernel_run import KernelRun
from kneemri.models.backbone import BASE, ENCODERS, SMALL, V3_BASE, V3_SMALL


class _Package:
    """A stand-in for a mounted weights package: just the members list the chooser reads."""

    def __init__(self, holdouts: list[float]) -> None:
        self.members = [type("M", (), {"holdout": score, "identifier": str(score)})() for score in holdouts]

    def strongest(self, count: int) -> list[object]:
        """Best holdout first, as MemberPackage does."""
        return sorted(self.members, key=lambda member: member.holdout, reverse=True)[:count]

    def named(self, wanted: list[str]) -> list[object]:
        """Exactly these, as MemberPackage does."""
        available = {member.identifier: member for member in self.members}
        return [available[name] for name in wanted]


def test_chosen():
    """Three selections: everything, the strongest few, and the ones an entry NAMES.

    Naming wins over a count when both are given, because the joint-holdout choice is the one an
    offline number was computed for and a solo ranking is not the same set (0.8612 against 0.8588 on
    our own holdout).
    """
    package = _Package([0.81, 0.87, 0.84])
    assert len(KernelRun().chosen(package)) == 3
    assert [member.holdout for member in KernelRun(members=1).chosen(package)] == [0.87]
    assert [member.holdout for member in KernelRun(names=["0.81", "0.84"]).chosen(package)] == [0.81, 0.84]
    assert [member.holdout for member in KernelRun(members=1, names=["0.84"]).chosen(package)] == [0.84]


def test_served():
    """An entry's views and vote survive to the ensemble, and the budget is attached for it.

    Both halves matter. The two axes an entry can vary WITHOUT retraining anything were reachable from
    `member_curve` and from nothing that writes a submission, so their board transfer could not be
    asked about at all; and a kernel that declared either used to lose the wall-clock ceiling with it,
    which is a nine-hour kill rather than a wrong number.
    """
    default = KernelRun().served()
    assert default.budget is not None
    assert default.vote is None and default.views is None, "the shipped entry declares neither"

    views, vote = SpatialViews(), LogitAverage()
    declared = KernelRun(serving=Serving(views=views, vote=vote)).served()
    assert declared.views is views
    assert declared.vote is vote
    assert declared.budget is not None, "declaring a vote must not cost the deadline"


def test_served_states_the_serving_in_the_log(caplog: pytest.LogCaptureFixture):
    """The run's own log names the vote, because nothing downstream of the kernel can.

    A submission's serving is invisible after the fact: the score comes back as one number and the
    entry's own log is the only record of what produced it. The first `kneemri-vote` run logged the
    kit, the members and the study count and never said which rule combined them — a slot spent on a
    flag whose landing could not be asserted.
    """
    with caplog.at_level("INFO"):
        KernelRun().served()
    assert "RankAverage (default)" in caplog.text, "the default is stated, not left silent"

    caplog.clear()
    with caplog.at_level("INFO"):
        KernelRun(serving=Serving(vote=LogitAverage(), views=SpatialViews())).served()
    assert "LogitAverage" in caplog.text
    assert "SpatialViews" in caplog.text


def test_decode():
    """The log must say what the members DID with `overlap`, not what the entry asked for.

    A whole-stack member never reads `overlap` — `Member.window` gives it one centred window — so on
    the current all-whole-stack pool the flag is inert and two entries differing only in it predict
    identically. The old line printed the request, and that difference was read as a confound in the
    board's member-count curve. Inert has to be SAID, not inferred.
    """
    whole = [type("M", (), {"whole_stack": True, "identifier": "ws"})()]
    mixed = [*whole, type("M", (), {"whole_stack": False, "identifier": "win"})()]

    assert "INERT" in KernelRun(overlap=True).decode(whole)
    assert "INERT" in KernelRun(overlap=False).decode(whole)

    honoured = KernelRun(overlap=True).decode(mixed)
    assert "INERT" not in honoured
    assert "1 of 2" in honoured and "win" in honoured, "only the windowed member honours the flag"


def test_find_encoder(tmp_path: Path):
    """The mount is matched on the ENCODER, not merely found, across both families at once.

    Mounting several and taking whichever sorts first would build a member at the wrong width, which
    surfaces as a shape error during load if we are lucky and as a silently wrong model if not. All
    four mounts are present together because that is the case the old `"dinov2" in path` predicate got
    wrong: it could not see a DINOv3 at all, and `vits16`/`vitb16` are not what our variants are called.
    """
    for parts in (
        ("models", "metaresearch", "dinov2", "PyTorch", "small", "1"),
        ("models", "metaresearch", "dinov2", "PyTorch", "base", "1"),
        ("datasets", "dinov3-hf", "vits16"),
        ("datasets", "dinov3-hf", "vitb16"),
    ):
        mount = tmp_path.joinpath(*parts)
        mount.mkdir(parents=True)
        (mount / "config.json").write_text("{}", encoding="utf-8")
    run = KernelRun(mounts=Mounts(root=tmp_path))

    assert run.find_encoder(ENCODERS[SMALL]).name == "1"
    assert "small" in str(run.find_encoder(ENCODERS[SMALL]))
    assert "base" in str(run.find_encoder(ENCODERS[BASE]))
    assert run.find_encoder(ENCODERS[V3_SMALL]).name == "vits16"
    assert run.find_encoder(ENCODERS[V3_BASE]).name == "vitb16"


def test_find_encoder_refuses_when_the_variant_is_not_mounted(tmp_path: Path):
    """A missing mount is fatal rather than silently substituted with the other size."""
    mount = tmp_path / "models" / "dinov2" / "small" / "1"
    mount.mkdir(parents=True)
    (mount / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        KernelRun(mounts=Mounts(root=tmp_path)).find_encoder(ENCODERS[BASE])


def test_find_encoder_refuses_when_two_mounts_match(tmp_path: Path):
    """Two directories answering to one encoder is unresolvable, and picking one loads wrong weights."""
    for version in ("1", "2"):
        mount = tmp_path / "models" / "metaresearch" / "dinov2" / "PyTorch" / "small" / version
        mount.mkdir(parents=True)
        (mount / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="matches 2 mounts"):
        KernelRun(mounts=Mounts(root=tmp_path)).find_encoder(ENCODERS[SMALL])


def test_run(monkeypatch: MonkeyPatch):
    """With no weights mounted the run must die, not write a submission from an untrained model."""
    monkeypatch.setattr(kernel_run.MemberPackage, "find", classmethod(lambda cls, mounts=None: None))
    with pytest.raises(FileNotFoundError):
        KernelRun().run()


def test_package(monkeypatch: MonkeyPatch):
    """The mount is resolved through one overridable seam, and its absence is fatal rather than empty.

    The seam exists so the fusion entry can merge two mounted packages; the contract that matters to
    every entry is the second half — an unmounted package must raise, because a run that quietly
    continues writes a submission from an untrained model and nothing downstream can tell.
    """
    sentinel = object()
    monkeypatch.setattr(kernel_run.MemberPackage, "find", classmethod(lambda cls, mounts=None: sentinel))
    assert KernelRun().package() is sentinel

    monkeypatch.setattr(kernel_run.MemberPackage, "find", classmethod(lambda cls, mounts=None: None))
    with pytest.raises(FileNotFoundError):
        KernelRun().package()


def test_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The kit's stamped commit, and an explicit UNSTAMPED when the kit predates the stamp.

    Unknown is reported rather than guessed: a kit whose version cannot be named is exactly the state
    that spent a submission on 2026-08-13, when the mounted source was 54 commits behind HEAD.

    The module sits at the depth the KIT puts it at — `kneemri_src/kneemri/kaggle/kernel_run.py` — because
    this test used to flatten that by one directory, which made a two-`.parent` lookup pass here while
    every real kernel run logged UNSTAMPED against a stamped kit.
    """
    source = tmp_path / "kneemri_src"
    (source / "kneemri" / "kaggle").mkdir(parents=True)
    monkeypatch.setattr(kernel_run, "__file__", str(source / "kneemri" / "kaggle" / "kernel_run.py"))

    assert "UNSTAMPED" in KernelRun().revision()

    (source / "kit.json").write_text(json.dumps({"commit": "abc1234", "dirty": False}), encoding="utf-8")
    assert KernelRun().revision() == "abc1234"

    (source / "kit.json").write_text(json.dumps({"commit": "abc1234", "dirty": True}), encoding="utf-8")
    assert KernelRun().revision() == "abc1234 DIRTY"
