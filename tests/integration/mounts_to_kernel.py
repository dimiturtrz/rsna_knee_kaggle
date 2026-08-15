"""The kernel's own loop: a mounted input tree becomes a loaded, checked ensemble.

`KernelRun.run` is the code a submission slot is spent on, and every line of it was written after a
failure that cost one. A DINOv3 member whose weights did not match the class the runtime built killed a
run with twenty-five other members loaded and ready, which is why an unloadable member is now DROPPED.
A run with nothing left must still refuse rather than write an empty table. And the entry's reading —
which members, overlapping windows or disjoint ones — is the difference between the two leaderboards.

None of it was executed by anything: `Mounts`, `MemberPackage.load` and `Encoder.locates` each have
unit tests, and the joint between them is what has failed every time.

`Submission` is where this stops. It decodes DICOM studies, so exercising it needs the competition data
rather than a synthetic tree — `checkpoint_to_submission` picks the chain up from there. Here it is
stubbed, and what it was HANDED is asserted, because that is the part `run` decides.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest
import torch

from core.mounts import Mounts
from kneemri.evaluation.member_record import MemberRecord
from kneemri.inference.ensemble import Ensemble
from kneemri.kaggle import kernel_run as kernel_module
from kneemri.kaggle.kernel_run import KernelRun
from kneemri.kaggle.package_members import MemberManifest, PackagedMember
from kneemri.models.backbone import ENCODERS, SMALL
from kneemri.models.model import StudyModel
from kneemri.training.train import TrainingOptions

from ..unit.kneemri.models.model import DIM, SLOTS, _FakeBackbone

SIZE = 28
SLICES = 4
SEED = 2026


@dataclass
class _Handed:
    """What `run` handed the submission it never got to write."""

    ensemble: Ensemble | None = None
    overlap: bool | None = None

    def stub(self, ensemble: Ensemble, competition: Path, working: Path, *, overlap: bool) -> "_Handed":
        """Stand in for `Submission`, recording the entry's reading instead of decoding studies."""
        self.ensemble = ensemble
        self.overlap = overlap
        assert competition.name, "the competition mount is what the sample submission is found beside"
        assert working, "a submission is written under the kernel's working directory"
        return self

    def run(self) -> object:
        """`Submission.run` returns the frame the kernel logs; nothing here reads it."""
        import pandas as pd  # noqa: PLC0415 - only the stub's return value needs it

        return pd.DataFrame({"StudyInstanceUID": []})


def _member(root: Path, identifier: str, holdout: float) -> PackagedMember:
    """One checkpoint, written by the same assembler a run writes with."""
    torch.manual_seed(SEED)
    model = StudyModel(_FakeBackbone(), DIM, SLOTS)
    options = TrainingOptions(image_size=SIZE, slices=SLICES)
    encoder = replace(ENCODERS[SMALL], dim=DIM)
    record = MemberRecord(options, encoder, model, torch.device("cpu"), [f"S{index}" for index in range(SLOTS)])
    payload = record.payload(model.state_dict(), holdout=holdout, epoch=0)
    path = root / f"member_{identifier}.pt"
    torch.save(payload, path)
    return PackagedMember(path, payload)


def _break(path: Path) -> None:
    """Corrupt a member's weights the way an encoder mismatch does: shapes that will not load."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    weights = cast(dict[str, torch.Tensor], payload["model"])
    name = next(iter(weights))
    weights[name] = torch.zeros(3, 5)
    torch.save(payload, path)


@pytest.fixture
def mounted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A `/kaggle/input` tree: a weights package, an encoder mount, the competition's sample table."""
    package = tmp_path / "kneemri-kit"
    package.mkdir()
    members = [_member(package, "fold0", 0.88), _member(package, "fold1", 0.80)]
    (package / "manifest.json").write_text(MemberManifest(members).document(), encoding="utf-8")

    # The path IS the recognition: `Encoder.locates` wants its family in the path and its leaf as a
    # whole segment, which is what keeps `vits16` from claiming `vits16plus`'s mount.
    weights = tmp_path / "models" / "metaresearch" / "dinov2" / "PyTorch" / "small" / "1"
    weights.mkdir(parents=True)
    (weights / "config.json").write_text(json.dumps({"model_type": "dinov2"}), encoding="utf-8")

    competition = tmp_path / "competition"
    competition.mkdir()
    (competition / "sample_submission.csv").write_text("StudyInstanceUID\n", encoding="utf-8")

    monkeypatch.setattr(kernel_module, "ENCODERS", {SMALL: replace(ENCODERS[SMALL], dim=DIM)})
    monkeypatch.setattr("transformers.AutoModel.from_pretrained", lambda weights: _FakeBackbone())
    # The kernel takes the card whenever there is one, and this suite never trains or serves on it.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return tmp_path


def _entry(mounted: Path, handed: _Handed, monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> KernelRun:
    """A kernel entry pointed at the synthetic tree, with the submission stubbed at the boundary."""
    monkeypatch.setattr(kernel_module, "Submission", handed.stub)
    return KernelRun(mounts=Mounts(root=mounted), **kwargs)  # type: ignore[arg-type]


def test_every_mounted_member_reaches_the_ensemble(mounted: Path, monkeypatch: pytest.MonkeyPatch):
    """The accuracy entry runs the whole package under overlapping windows.

    The fingerprint check runs inside `MemberPackage.load`, so a member arriving here is one whose
    weights were rebuilt into the architecture its own config describes.
    """
    handed = _Handed()
    _entry(mounted, handed, monkeypatch).run()

    ensemble = handed.ensemble
    assert ensemble is not None
    assert sorted(ensemble.models) == ["fold0", "fold1"]
    assert handed.overlap is True, "overlapping windows are worth +0.0156, the strongest number here"


def test_the_efficiency_entry_takes_the_strongest_member_only(mounted: Path, monkeypatch: pytest.MonkeyPatch):
    """One member, disjoint windows — the other leaderboard, which rewards the opposite things."""
    handed = _Handed()
    _entry(mounted, handed, monkeypatch, members=1, overlap=False).run()

    ensemble = handed.ensemble
    assert ensemble is not None
    assert list(ensemble.models) == ["fold0"], "strongest by recorded holdout, which is the only ranking there is"
    assert handed.overlap is False


def test_a_member_that_will_not_load_is_dropped_rather_than_fatal(mounted: Path, monkeypatch: pytest.MonkeyPatch):
    """The failure this loop exists for: one bad member must not take the entry down with it.

    A DINOv3 member cost a whole submission this way — its weights did not match the class the runtime
    built, and the run died with every other member already loaded.
    """
    _break(mounted / "kneemri-kit" / "member_fold1.pt")
    handed = _Handed()
    _entry(mounted, handed, monkeypatch).run()

    ensemble = handed.ensemble
    assert ensemble is not None
    assert list(ensemble.models) == ["fold0"], "the entry runs on what loaded"


def test_an_entry_with_nothing_loaded_refuses(mounted: Path, monkeypatch: pytest.MonkeyPatch):
    """Better a raise than a submission written from no members at all."""
    for name in ("member_fold0.pt", "member_fold1.pt"):
        _break(mounted / "kneemri-kit" / name)
    handed = _Handed()

    with pytest.raises(RuntimeError, match="no member loaded"):
        _entry(mounted, handed, monkeypatch).run()

    assert handed.ensemble is None, "nothing may reach the submission when nothing loaded"


def test_an_unmounted_package_is_named_rather_than_guessed(mounted: Path, monkeypatch: pytest.MonkeyPatch):
    """A kernel with no weights attached fails at the mount, before it spends the slot decoding."""
    (mounted / "kneemri-kit" / "manifest.json").unlink()
    handed = _Handed()

    with pytest.raises(FileNotFoundError, match="no weights package mounted"):
        _entry(mounted, handed, monkeypatch).run()
