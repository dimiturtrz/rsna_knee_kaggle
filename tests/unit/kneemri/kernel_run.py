"""What a kernel does once it can import us — without a Kaggle mount."""

from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch

from kneemri import kernel_run
from kneemri.kernel_run import KernelRun


class _Package:
    """A stand-in for a mounted weights package: just the members list the chooser reads."""

    def __init__(self, holdouts: list[float]) -> None:
        self.members = [type("M", (), {"holdout": score, "identifier": str(score)})() for score in holdouts]

    def strongest(self, count: int) -> list[object]:
        """Best holdout first, as MemberPackage does."""
        return sorted(self.members, key=lambda member: member.holdout, reverse=True)[:count]


def test_chosen():
    """The accuracy entry runs everything; the efficiency entry runs the strongest few."""
    package = _Package([0.81, 0.87, 0.84])
    assert len(KernelRun().chosen(package)) == 3
    assert [member.holdout for member in KernelRun(members=1).chosen(package)] == [0.87]


def test_find_dinov2(tmp_path: Path, monkeypatch: MonkeyPatch):
    """The mount is matched on VARIANT, not merely found.

    Mounting both sizes and taking whichever sorts first would build a member at the wrong width, which
    surfaces as a shape error during load if we are lucky and as a silently wrong model if not.
    """
    for variant in ("base", "small"):
        mount = tmp_path / "models" / "metaresearch" / "dinov2" / "PyTorch" / variant / "1"
        mount.mkdir(parents=True)
        (mount / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(kernel_run, "INPUT", tmp_path)

    assert KernelRun().find_dinov2("small").name == "1"
    assert "small" in str(KernelRun().find_dinov2("small"))
    assert "base" in str(KernelRun().find_dinov2("base"))


def test_find_dinov2_refuses_when_the_variant_is_not_mounted(tmp_path: Path, monkeypatch: MonkeyPatch):
    """A missing mount is fatal rather than silently substituted with the other size."""
    mount = tmp_path / "models" / "dinov2" / "small" / "1"
    mount.mkdir(parents=True)
    (mount / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(kernel_run, "INPUT", tmp_path)

    with pytest.raises(FileNotFoundError):
        KernelRun().find_dinov2("base")


def test_run(monkeypatch: MonkeyPatch):
    """With no weights mounted the run must die, not write a submission from an untrained model."""
    monkeypatch.setattr(kernel_run.MemberPackage, "find", classmethod(lambda cls: None))
    with pytest.raises(FileNotFoundError):
        KernelRun().run()


def test_package(monkeypatch: MonkeyPatch):
    """The mount is resolved through one overridable seam, and its absence is fatal rather than empty.

    The seam exists so the fusion entry can merge two mounted packages; the contract that matters to
    every entry is the second half — an unmounted package must raise, because a run that quietly
    continues writes a submission from an untrained model and nothing downstream can tell.
    """
    sentinel = object()
    monkeypatch.setattr(kernel_run.MemberPackage, "find", classmethod(lambda cls: sentinel))
    assert KernelRun().package() is sentinel

    monkeypatch.setattr(kernel_run.MemberPackage, "find", classmethod(lambda cls: None))
    with pytest.raises(FileNotFoundError):
        KernelRun().package()
