"""The generated command index, over a package this test builds itself."""

from pathlib import Path

import pytest

from kneemri.commands import Commands, main


def _package(root: Path) -> Path:
    """A miniature package with one command, one plain module, and one command in a subpackage."""
    package = root / "fake"
    (package / "corpus").mkdir(parents=True)
    (package / "tool.py").write_text(
        '"""Answer a question: `python -m fake.tool`.\n\nMore prose.\n"""\n\ndef main():\n    pass\n'
    )
    (package / "library.py").write_text('"""Not a command."""\n\ndef helper():\n    pass\n')
    (package / "corpus" / "probe.py").write_text('"""Probe the corpus."""\n\ndef main():\n    pass\n')
    return package


def test_entries(tmp_path: Path):
    """Only modules defining a top-level `main` are commands, and the summary is their own first line.

    The plain module is the case that matters: `helper` is a top-level function too, so a scan that
    looked for any definition rather than for `main` would list every module in the package.
    """
    entries = Commands(_package(tmp_path)).entries()

    assert [module for module, _ in entries] == ["fake.corpus.probe", "fake.tool"], "sorted, and no library"
    assert dict(entries)["fake.tool"] == "Answer a question: `python -m fake.tool`."
    assert "More prose" not in dict(entries)["fake.tool"], "the FIRST line only, or the table grows paragraphs"


def test_rendered(tmp_path: Path):
    """The index groups by subpackage and carries every command as an invocation."""
    rendered = Commands(_package(tmp_path)).rendered()

    assert "### corpus" in rendered
    assert "| `python -m fake.corpus.probe` | Probe the corpus. |" in rendered
    assert "| `python -m fake.tool` |" in rendered
    assert rendered.startswith("# kneemri — the commands")


def test_stale(tmp_path: Path):
    """A missing index is stale, a written one is not, and a NEW command makes it stale again.

    The third is the whole point of the check: the failure being guarded is a command that lands with no
    row, which is exactly how fourteen of them became invisible.
    """
    package = _package(tmp_path)
    commands = Commands(package)
    index = tmp_path / "README.md"

    assert commands.stale(index), "an index that does not exist yet cannot be current"

    commands.write(index)
    assert not commands.stale(index)

    (package / "later.py").write_text('"""Added after the index was written."""\n\ndef main():\n    pass\n')
    assert commands.stale(index), "a new command must make the index stale"


def test_write(tmp_path: Path):
    """Writing produces exactly what rendering says, so the check cannot disagree with the generator."""
    commands = Commands(_package(tmp_path))
    index = tmp_path / "README.md"
    commands.write(index)

    assert index.read_text(encoding="utf-8") == commands.rendered()


def test_main_check_fails_on_a_stale_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """`--check` exits nonzero rather than quietly rewriting, since CI must fail on drift."""
    package = _package(tmp_path)
    index = tmp_path / "README.md"
    monkeypatch.setattr("sys.argv", ["commands", "--check", "--index", str(index), "--package", str(package)])

    with pytest.raises(SystemExit, match="out of date"):
        main()
