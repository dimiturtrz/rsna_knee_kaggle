"""A converted snapshot's location, which is the only half of it a unit test may touch.

`build` loads a gigabyte of real weights off a machine-local path — no unit test may reach that, and
`registry_to_checkpoints` already holds every registry entry to its own snapshot. What is testable here
is the seam that decides WHERE it looks, and that seam has a real failure available to it: a mounted
source silently ignored would send a kernel to this machine's data root, which does not exist there.
"""

from pathlib import Path

from kneemri.models.snapshot_trunk import ORTHOFOUNDATION_L, LocalSnapshot


def test_resolve(tmp_path: Path) -> None:
    """A given source wins outright, and it is returned as itself rather than joined to anything.

    The absolute case is what a Kaggle mount looks like, so it is the one that has to hold; it is spelled
    with `tmp_path` rather than a literal `/kaggle/input/...` because a unit test may name no path it did
    not create, and the assertion is about pass-through, not about the string.
    """
    snapshot = LocalSnapshot("somewhere/under_reference")
    mounted = tmp_path / "mounted"
    assert mounted.is_absolute()
    assert snapshot.resolve(str(mounted)) == mounted
    assert snapshot.resolve(Path("relative/mount")) == Path("relative/mount")


def test_orthofoundation_names_a_directory_under_reference() -> None:
    """The registry entry may not carry an absolute path — that is the whole reason this type exists."""
    assert not Path(ORTHOFOUNDATION_L.directory).is_absolute()
    assert ORTHOFOUNDATION_L.directory.endswith("hf_snapshot")
