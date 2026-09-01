"""This builder's COMMAND LINE, exercised through the argv it actually parses.

`--help` reaches parser construction and exits before any filesystem work, which is where the defect
this class of test exists for lives: `build_rijeka` was uncallable for as long as `CacheFlags` owned a
`--spacing` flag, because argparse raised while CONSTRUCTING the parser. See `build_cache`'s own test.
"""

import pytest

from core.data.corpus.build_mrnet_labels import main


def test_main(monkeypatch: pytest.MonkeyPatch):
    """The parser CONSTRUCTS, which is the step a flag collision dies in."""
    monkeypatch.setattr("sys.argv", ["build_mrnet_labels", "--help"])

    with pytest.raises(SystemExit) as exit_code:
        main()

    assert exit_code.value.code == 0
