import argparse
from pathlib import Path

import pytest

from core.paths import ENV_OVERRIDE, DataRoot
from core.roster_flags import RosterFlags


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataRoot:
    monkeypatch.delenv(ENV_OVERRIDE, raising=False)
    config = tmp_path / "paths.yaml"
    config.write_text(f"data: {(tmp_path / 'root').as_posix()}\n", encoding="utf-8")
    return DataRoot.from_config(config)


@pytest.fixture
def parser() -> argparse.ArgumentParser:
    declared = argparse.ArgumentParser()
    RosterFlags.add(declared)
    return declared


def test_add():
    declared = argparse.ArgumentParser()
    RosterFlags.add(declared)
    parsed = declared.parse_args([])
    assert (parsed.roster, parsed.candidates) == (None, [])


def test_checkpoints(parser: argparse.ArgumentParser, root: DataRoot, tmp_path: Path):
    roster = tmp_path / "members.roster"
    roster.write_text("# header\n\nalpha.pt   # why\n  beta  \nelsewhere/gamma.pt\n", encoding="utf-8")
    expected = [root.models() / "alpha.pt", root.models() / "beta.pt", Path("elsewhere/gamma.pt")]
    assert RosterFlags.checkpoints(parser.parse_args(["--roster", str(roster)]), root) == expected
    from_flags = parser.parse_args(["--candidates", "alpha", "beta.pt", "elsewhere/gamma.pt"])
    assert RosterFlags.checkpoints(from_flags, root) == expected


def test_required(parser: argparse.ArgumentParser, root: DataRoot):
    named = parser.parse_args(["--candidates", "alpha"])
    assert RosterFlags.required(named, root) == [root.models() / "alpha.pt"]
    with pytest.raises(SystemExit):
        RosterFlags.required(parser.parse_args([]), root)
