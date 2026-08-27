"""The corpus builders' COMMAND LINES, exercised through the argv they actually parse.

WHY `--help` IS THE TEST. `build_rijeka` was uncallable for as long as `CacheFlags` has owned a
`--spacing` flag: the builder declared its own float `--spacing` for millimetres-per-pixel, argparse
raised `conflicting option string` while CONSTRUCTING the parser, and the CLI died before it could read
one argument. Nothing caught it, because no test ever built these parsers — the modules are shells whose
only untested line was the one that broke. `--help` reaches parser construction and exits before any
filesystem work, so it is the whole of the defect and none of the decode.
"""

from importlib import import_module

import pytest

from core.data.corpus.build_cache import main
from core.data.corpus.slots import FOREIGN_SCHEMES

BUILDERS = ("build_cache", "build_merged", "build_mrnet", "build_rijeka")


@pytest.mark.parametrize("builder", BUILDERS)
def test_main(builder: str, monkeypatch: pytest.MonkeyPatch):
    """Every builder's parser CONSTRUCTS, which is the step `--spacing` collided in."""
    monkeypatch.setattr("sys.argv", [builder, "--help"])

    with pytest.raises(SystemExit) as exit_code:
        import_module(f"core.data.corpus.{builder}").main()

    assert exit_code.value.code == 0, f"{builder} cannot build its own parser"


@pytest.mark.parametrize("scheme", FOREIGN_SCHEMES)
def test_foreign_scheme_refused(scheme: str, monkeypatch: pytest.MonkeyPatch):
    """`build_cache` decodes OUR study list whatever `--scheme` says, so a foreign name is refused.

    `--scheme` selects a FILENAME and nothing else consults it, and for `merged` the shape matches ours
    exactly — six slots either way — so neither the log line nor the array can catch the mistake. Only
    the refusal can. It cost 59.7 GB (knee-48br).
    """
    monkeypatch.setattr("sys.argv", ["build_cache", "--scheme", scheme])

    with pytest.raises(SystemExit) as exit_code:
        main()

    assert exit_code.value.code == 2, f"--scheme {scheme} must be refused, not decoded under our corpus"
