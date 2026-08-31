"""The feature pass over our split: what it writes, what it leaves NaN, and the teacher it refuses."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from jaxtyping import Float
from numpy.typing import NDArray

from core.findings import STUDY
from kneemri.evaluation.raptor_feed import RaptorFeed
from kneemri.evaluation.raptor_model import RaptorMember
from kneemri.kaggle.raptor_features import FeaturePass
from kneemri.kaggle.raptor_teacher import TRAIN
from kneemri.kaggle.submission import Split
from kneemri.teacher_file import TeacherFile

WIDTH = 3
DEPTH = 2
SIDE = 4


class _Reader:
    """A member-shaped stub: it answers a width and reads a study, or refuses to read one."""

    def __init__(self, value: float, *, refuses: bool = False) -> None:
        self.value = value
        self.refuses = refuses

    def width(self) -> int:
        """The teacher width the pass sizes its rows from, never a default."""
        return WIDTH

    def represented(
        self,
        volume: Float[NDArray[np.uint8], "depth height width"],
        mask: Float[NDArray[np.uint8], " depth"],
        device: str,
    ) -> Float[NDArray[np.float32], " features"]:
        """A constant reading, or the failure a real member raises on a study it cannot window."""
        del mask, device
        if self.refuses:
            message = "this member cannot read that study"
            raise ValueError(message)
        return np.full(WIDTH, self.value + float(volume.sum()), dtype=np.float32)


def _corpus(root: Path, studies: list[str]) -> Path:
    """A split laid out the way the competition download is: a table and a series table beside it."""
    (root / f"{TRAIN}.csv").write_text(f"{STUDY}\n" + "\n".join(studies) + "\n", encoding="utf-8")
    rows = "\n".join(f"{study},{study}-a" for study in studies)
    (root / f"{TRAIN}_series.csv").write_text(f"{STUDY},SeriesInstanceUID\n{rows}\n", encoding="utf-8")
    return root


def _members(**readers: _Reader) -> dict[str, RaptorMember]:
    """The stubs under the names the written files are keyed by."""
    return {name: cast("RaptorMember", reader) for name, reader in readers.items()}


def test_embeddings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every member's reading of every study, from ONE decode each, in the split's own order.

    The rows are sized from the MEMBER's width rather than from a constant, because a teacher file is
    joined on nothing but its study order and a row of the wrong width would only surface when a student
    built a projection into it.
    """
    monkeypatch.setattr(
        RaptorFeed,
        "study",
        lambda self, study, records: (np.full((DEPTH, SIDE, SIDE), 1, np.uint8), np.ones(DEPTH, np.uint8)),
    )
    reading = FeaturePass(_corpus(tmp_path, ["a", "b"]), tmp_path / "out.npz", Split(name=TRAIN))

    rows = reading.embeddings(["a", "b"], reading.records(), _members(one=_Reader(0.5)), "cpu")

    assert list(rows) == ["one"]
    assert rows["one"].shape == (2, WIDTH)
    assert not np.isnan(rows["one"]).any()


def test_embeddings_leaves_nan_where_nothing_was_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A study that will not decode, and a member that will not read it, both leave NaN and carry on.

    Stopping would throw away thirty minutes of decode over one unreadable study, and a fallback value
    would put a direction in the file that no member ever read — which `TeacherFeatures.rows` could then
    never tell from a reading. NaN is the only value that stays loud all the way to the batch.
    """

    def _refuse(self: RaptorFeed, study: str, records: object) -> tuple[object, object]:
        del self, records
        message = f"{study} will not decode"
        raise ValueError(message)

    monkeypatch.setattr(RaptorFeed, "study", _refuse)
    reading = FeaturePass(_corpus(tmp_path, ["a"]), tmp_path / "out.npz", Split(name=TRAIN))
    undecodable = reading.embeddings(["a"], reading.records(), _members(one=_Reader(0.5)), "cpu")

    assert np.isnan(undecodable["one"]).all()

    monkeypatch.setattr(
        RaptorFeed,
        "study",
        lambda self, study, records: (np.zeros((DEPTH, SIDE, SIDE), np.uint8), np.ones(DEPTH, np.uint8)),
    )
    mixed = reading.embeddings(
        ["a"], reading.records(), _members(good=_Reader(0.5), bad=_Reader(0.5, refuses=True)), "cpu"
    )

    assert not np.isnan(mixed["good"]).any(), "one member's failure must not cost the other its reading"
    assert np.isnan(mixed["bad"]).all()


def test_wrote(tmp_path: Path) -> None:
    """The representations land under `embeddings`, which is NOT the name the targets file uses.

    A shared name would let `--teacher-features raptor_teacher.npz` load twelve probabilities as a
    twelve-wide 'embedding' and report a representation arm that had distilled logits.
    """
    destination = tmp_path / "fleet" / "features.npz"
    embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    reading = FeaturePass(tmp_path, destination, Split(name=TRAIN))

    written = reading.wrote(destination, ["a", "b"], embeddings)
    stored = np.load(written)

    assert written.exists(), "the fleet directory is made rather than assumed"
    assert list(stored[TeacherFile.IDENTIFIERS]) == ["a", "b"]
    assert np.allclose(stored[TeacherFile.EMBEDDINGS], embeddings)
    assert TeacherFile.SCORES not in stored


def test_honest(tmp_path: Path) -> None:
    """A teacher of OURS fitted on every study is refused; anything it cannot read passes.

    A `--fold -1` member has read the holdout its student would be scored on, so its readings there are
    memorised and the student's number would flatter both. A foreign checkpoint records no recipe, so it
    passes here — which is not a clean bill of health, only the limit of what a file can be asked.
    """
    full = tmp_path / "full.pt"
    torch.save({"recipe": {"fold": -1}}, full)
    honest = tmp_path / "fold0.pt"
    torch.save({"recipe": {"fold": 0}}, honest)
    foreign = tmp_path / "foreign.pt"
    torch.save({"model": {}}, foreign)

    with pytest.raises(ValueError, match="fold -1"):
        FeaturePass.honest(full)

    assert FeaturePass.honest(honest) is None
    assert FeaturePass.honest(foreign) is None
