"""KMAR's builder command line, exercised through the argv it actually parses.

Same argument as `build_fastmri`'s tests: this module is a shell, and its only untested lines are the
two that can destroy something — the parser it constructs, and the scheme it refuses.
"""

from pathlib import Path

import pytest

from core.data.corpus.build_kmar import main
from core.data.corpus.slots import CANONICAL_SCHEME, KMAR_SCHEME


def test_main(monkeypatch: pytest.MonkeyPatch):
    """The parser CONSTRUCTS — the step a flag colliding with `CacheFlags` dies at."""
    monkeypatch.setattr("sys.argv", ["build_kmar", "--help"])

    with pytest.raises(SystemExit) as exit_code:
        main()

    assert exit_code.value.code == 0, "build_kmar cannot build its own parser"


def test_main_refuses_our_own_scheme(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Anything but `kmar` is refused, and the refusal is checked by its MESSAGE, not its exit code.

    `Routing.suffix()` is empty for the canonical scheme, so a KMAR decode under it overwrites the
    71.6 GB of arrays every member on disk was fitted on — with a corpus differing only in axis 0, which
    nothing downstream could detect. Exit code 2 alone would not prove the guard fired: argparse exits 2
    for an unparseable flag as well, so a test asserting only the code would pass with the guard deleted.
    """
    monkeypatch.setattr(
        "sys.argv",
        ["build_kmar", "--archive", str(tmp_path), "--scheme", CANONICAL_SCHEME],
    )

    with pytest.raises(SystemExit) as exit_code:
        main()

    assert exit_code.value.code == 2
    assert f"--scheme {KMAR_SCHEME} is required" in capsys.readouterr().err, "the guard did not fire"
