"""A checkpoint's self-description is read, never guessed — and a file that is not one says so."""

from pathlib import Path

import torch

from core.findings import TARGETS
from kneemri.evaluation.raptor_inventory import CheckpointEntry, Inventory
from kneemri.evaluation.raptor_model import DEFAULT_ARCHITECTURE, DEFAULT_RESOLUTION

FOREIGN = "convnextv2_base.fcmae_ft_in22k_in1k_384"


def saved(path: Path, **state: object) -> Path:
    """Write a checkpoint carrying exactly the keys a case is about, and nothing else."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return path


def test_read(tmp_path: Path) -> None:
    """A checkpoint that names its backbone is read by that name, and one that does not takes the arm's.

    The two classes are the whole input space here: the published arm ships no `arch` key, and every
    later dataset does. An inventory that defaulted differently from `RaptorMember.loaded` would report
    a fleet the serving path cannot build.
    """
    named = CheckpointEntry.read(saved(tmp_path / "named.pt", model={}, arch=FOREIGN, res=336))
    assert (named.architecture, named.resolution, named.loadable) == (FOREIGN, 336, True)

    bare = CheckpointEntry.read(saved(tmp_path / "bare.pt", model={}))
    assert (bare.architecture, bare.resolution) == (DEFAULT_ARCHITECTURE, DEFAULT_RESOLUTION)
    assert bare.labels == TARGETS

    stranger = CheckpointEntry.read(saved(tmp_path / "stranger.pt", state_dict={}))
    assert stranger.loadable is False


def test_family(tmp_path: Path) -> None:
    """Resolution is part of the family: one backbone at two sizes is two lenses, not one repeated."""
    small = CheckpointEntry.read(saved(tmp_path / "small.pt", model={}, arch=FOREIGN, res=336))
    large = CheckpointEntry.read(saved(tmp_path / "large.pt", model={}, arch=FOREIGN, res=480))
    assert small.family != large.family
    assert small.family == f"{FOREIGN}@336"


def test_line(tmp_path: Path) -> None:
    """The row names the defect it is read for — an unservable file, or a label order that is not ours."""
    entry = CheckpointEntry.read(saved(tmp_path / "sub" / "c.pt", state_dict={}, lab=["ACL"]))
    line = entry.line(tmp_path)
    assert "NOT-A-RAPTOR-CHECKPOINT" in line
    assert "FOREIGN-LABEL-ORDER" in line
    assert line.endswith(str(Path("sub") / "c.pt"))


def test_scanned(tmp_path: Path) -> None:
    """Every `.pt` below the root is found, whatever directory the author's dataset title made."""
    saved(tmp_path / "one" / "a.pt", model={}, arch=FOREIGN, res=336)
    saved(tmp_path / "two" / "b.pt", model={})
    (tmp_path / "two" / "notes.txt").write_text("not a checkpoint")

    inventory = Inventory.scanned(tmp_path)
    assert len(inventory.entries) == 2


def test_families(tmp_path: Path) -> None:
    """The census counts loadable checkpoints only, so a stranger cannot inflate the published arm's."""
    saved(tmp_path / "a.pt", model={}, arch=FOREIGN, res=336)
    saved(tmp_path / "b.pt", model={}, arch=FOREIGN, res=336)
    saved(tmp_path / "c.pt", state_dict={})

    assert Inventory.scanned(tmp_path).families() == {f"{FOREIGN}@336": 2}


def test_report(tmp_path: Path) -> None:
    """The report carries a row per checkpoint and a census that agrees with the rows above it."""
    saved(tmp_path / "a.pt", model={}, arch=FOREIGN, res=336)
    saved(tmp_path / "b.pt", state_dict={})

    lines = Inventory.scanned(tmp_path).report(tmp_path)
    assert "1 of 2 loadable, in 1 families:" in lines
    assert sum(1 for line in lines if line.startswith(("ok", "NOT-A-RAPTOR"))) == 2
