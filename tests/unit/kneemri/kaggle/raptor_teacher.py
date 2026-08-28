"""A foreign member's readings over our own split, without a checkpoint or a DICOM."""

from pathlib import Path

import numpy as np
import pytest

from core.findings import STUDY
from kneemri.evaluation.raptor_model import RaptorMember
from kneemri.kaggle.raptor_teacher import TRAIN, TeacherPass
from kneemri.kaggle.submission import Split
from kneemri.teacher_file import TeacherFile


def _corpus(root: Path, studies: list[str]) -> Path:
    """A split laid out the way the competition download is: a table and a series table beside it."""
    (root / f"{TRAIN}.csv").write_text(f"{STUDY}\n" + "\n".join(studies) + "\n", encoding="utf-8")
    rows = "\n".join(f"{study},{study}-a" for study in studies)
    (root / f"{TRAIN}_series.csv").write_text(f"{STUDY},SeriesInstanceUID\n{rows}\n", encoding="utf-8")
    return root


def test_identifiers(tmp_path: Path):
    """Every study of the split, as strings, in the table's own order.

    Strings because the identifiers are what a training run JOINS on, and a numeric-looking study read
    back as an int would miss every row of a file keyed by text — silently, as a teacher that taught
    nothing rather than as a crash.
    """
    teaching = TeacherPass(_corpus(tmp_path, ["1.2.3", "4.5.6"]), tmp_path / "out.npz")

    assert teaching.identifiers() == ["1.2.3", "4.5.6"]


def test_records(tmp_path: Path):
    """The series rows, grouped under the study they belong to."""
    teaching = TeacherPass(_corpus(tmp_path, ["1.2.3", "4.5.6"]), tmp_path / "out.npz")
    records = teaching.records()

    assert set(records) == {"1.2.3", "4.5.6"}
    assert records["1.2.3"][0]["SeriesInstanceUID"] == "1.2.3-a"


def test_entry(tmp_path: Path):
    """The scorer runs over the TRAIN split, not the competition's test set.

    The whole point of the pass is that it reads studies whose labels we hold; an entry left on the
    default split would score the test set and write a teacher file with no overlap at all.
    """
    teaching = TeacherPass(_corpus(tmp_path, ["1.2.3"]), tmp_path / "out.npz")
    entry = teaching.entry()

    assert entry.split.name == TRAIN
    assert entry.competition == tmp_path


def test_written(tmp_path: Path):
    """The readings land under the two names the training side looks them up by, identifiers included."""
    destination = tmp_path / "teacher.npz"
    scores = np.array([[0.1, 0.9], [0.4, 0.6]], dtype=np.float32)
    teaching = TeacherPass(tmp_path, destination, Split(name=TRAIN))
    written = teaching.written(destination, ["1.2.3", "4.5.6"], scores)
    read = np.load(written)

    assert written == destination
    assert list(read[TeacherFile.IDENTIFIERS]) == ["1.2.3", "4.5.6"]
    assert np.allclose(read[TeacherFile.SCORES], scores)

    nested = teaching.written(tmp_path / "fleet" / "v5.npz", ["1.2.3", "4.5.6"], scores)
    assert nested.exists()


def test_fleet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checkpoints are keyed by their filename STEM, which is what each teacher file is then named for.

    The key matters more than it looks: the distance report reads a member back by this name, so keying
    on an index into the argument list would make every report unreadable against the checkpoints.
    """
    monkeypatch.setattr(RaptorMember, "loaded", classmethod(lambda cls, path, device: (path, device)))

    members = TeacherPass.fleet([Path("a/raptor_ft_coatnet_v5_full_swa.pt"), Path("b/cnn336.pt")], "cpu")

    assert list(members) == ["raptor_ft_coatnet_v5_full_swa", "cnn336"]


def test_sharded():
    """Strided slices partition the split exactly — every study once, none twice.

    Exactly, because a teacher file is joined from these and never checked against the split again: a
    shard scheme that dropped a study would train the student on a fallback row it could not tell from
    a reading, and one that repeated a study would weight it twice.
    """
    studies = [f"s{index}" for index in range(7)]
    shards = [TeacherPass.sharded(studies, shard, 3) for shard in range(3)]

    assert shards[0] == ["s0", "s3", "s6"]
    assert sorted(sum(shards, [])) == sorted(studies)
