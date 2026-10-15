"""A foreign member's readings over our own split, without a checkpoint or a DICOM."""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch

from core import cli
from core.findings import STUDY
from core.paths import ENV_OVERRIDE, DataRoot
from kneemri.device import VisibleCards
from kneemri.evaluation.raptor_feed import NARTAA
from kneemri.evaluation.raptor_model import RaptorMember
from kneemri.kaggle import raptor_features, raptor_teacher
from kneemri.kaggle.raptor_run import RaptorEntry
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
    monkeypatch.setattr(RaptorMember, "loaded", classmethod(lambda cls, path, device, layout: (path, layout)))

    members = TeacherPass.fleet([Path("a/raptor_ft_coatnet_v5_full_swa.pt"), Path("b/cnn336.pt")], "cpu", NARTAA)

    assert list(members) == ["raptor_ft_coatnet_v5_full_swa", "cnn336"]
    assert members["cnn336"] == (Path("b/cnn336.pt"), NARTAA)


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


def test_honest(tmp_path: Path) -> None:
    """The logit pass refuses no teacher, not even one that names no file on disk."""
    assert TeacherPass.honest(tmp_path / "absent.pt") is None


def test_taught(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The logit pass takes the scorer's readings, unchanged."""
    readings = {"a": np.full((1, 2), 0.25)}
    monkeypatch.setattr(RaptorEntry, "readings", lambda self, identifiers, records, members, device: readings)
    teaching = TeacherPass(_corpus(tmp_path, ["1.2.3"]), tmp_path / "out.npz")

    assert teaching.taught(["1.2.3"], teaching.records(), {}, "cpu") is readings


def test_kept(tmp_path: Path) -> None:
    """The logit pass keeps its readings under `scores`, the name the targets loader asks for."""
    teaching = TeacherPass(tmp_path, tmp_path / "out.npz")
    kept = teaching.kept(tmp_path / "out.npz", ["1.2.3"], np.array([[0.5, 0.5]], np.float32))

    assert TeacherFile.SCORES in np.load(kept)


class _ParsedError(Exception):
    """Raised once the command line is read, so a test can stop a pass before it loads a member."""


def _namespace(monkeypatch: pytest.MonkeyPatch, command: Callable[[], None], argv: list[str]) -> argparse.Namespace:
    """What a pass command reads `argv` as, captured before any of it is acted on."""
    seen: list[argparse.Namespace] = []
    original = cli.Command.parsed

    def _capture(self: cli.Command) -> argparse.Namespace:
        seen.append(original(self))
        raise _ParsedError

    with monkeypatch.context() as patch:
        patch.setattr(sys, "argv", ["pass", *argv])
        patch.setattr(cli.Command, "parsed", _capture)
        with pytest.raises(_ParsedError):
            command()
    return seen[0]


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The command reads its flags, scores its shard through every member, and merges shards byte for byte.

    The feature pass is held to the same reading of the same flags, because it is the same command over
    the same split and an arm script switches between the two by module name alone.
    """
    argv = ["--weights", "a.pt", "--shard", "1", "--shards", "2"]
    expected_flags = argparse.Namespace(
        paths=cli.PATHS,
        weights=[Path("a.pt")],
        fleet=None,
        split=TRAIN,
        out=None,
        studies=0,
        shard=1,
        shards=2,
        merge=None,
        layout="raptor",
    )
    assert _namespace(monkeypatch, raptor_teacher.main, argv) == expected_flags
    assert _namespace(monkeypatch, raptor_features.main, argv) == expected_flags

    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    raw = DataRoot(tmp_path).raw()
    raw.mkdir(parents=True)
    _corpus(raw, ["s0", "s1", "s2", "s3"])
    monkeypatch.setattr(VisibleCards, "device", lambda self: torch.device("cpu"))
    monkeypatch.setattr(
        TeacherPass, "fleet", staticmethod(lambda weights, device, layout: {p.stem: device for p in weights})
    )
    monkeypatch.setattr(
        RaptorEntry,
        "readings",
        lambda self, identifiers, records, members, device: {
            name: np.full((len(identifiers), 2), float(index)) for index, name in enumerate(members)
        },
    )
    fleet = tmp_path / "fleet"
    argv = ["raptor_teacher", "--weights", "a.pt", "b.pt", "--fleet", str(fleet), "--shard", "1", "--shards", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    raptor_teacher.main()

    second = np.load(fleet / "b.npz")
    assert list(second[TeacherFile.IDENTIFIERS]) == ["s1", "s3"]
    assert second[TeacherFile.SCORES].dtype == np.float32
    assert np.allclose(second[TeacherFile.SCORES], 1.0)

    merged = tmp_path / "merged.npz"
    shards = [str(fleet / "a.npz"), str(fleet / "b.npz")]
    monkeypatch.setattr(sys, "argv", ["raptor_teacher", "--weights", "a.pt", "--out", str(merged), "--merge", *shards])
    raptor_teacher.main()

    expected = tmp_path / "expected.npz"
    values = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]], np.float32)
    TeacherFile(expected).write(["s1", "s3", "s1", "s3"], values)
    assert merged.read_bytes() == expected.read_bytes()
