"""The one constant that is changed in his program, and the three ways his table is refused.

Nothing here runs his program — it is 3,105 lines over four gigabytes of mounted weights. What is
testable is the part that can be wrong SILENTLY: a budget patch that lands on one of its two sites,
and a table that is read and blended when it should have been refused.
"""

from pathlib import Path

import pandas as pd
import pytest

from core.findings import STUDY, TARGETS
from kneemri.kaggle.bend_entry import BUDGET_SOURCE, FUTURE, SUBMISSION, WEIGHTS_SOURCE, BendEntry

HOUR = 3600.0


def _entry(tmp_path: Path, seconds: float = 2.0 * HOUR) -> BendEntry:
    """An entry over a throwaway source file and a throwaway working directory."""
    source = tmp_path / "all_public.py"
    source.write_text(f"{BUDGET_SOURCE}\nman = dict()\n{WEIGHTS_SOURCE}\n    pass\n{BUDGET_SOURCE}\n", encoding="utf-8")
    return BendEntry(source=source, working=tmp_path, seconds=seconds)


def _table(studies: int = 3) -> pd.DataFrame:
    """A well-formed foreign submission."""
    frame = pd.DataFrame({STUDY: [f"s{index}" for index in range(studies)]})
    for offset, target in enumerate(TARGETS):
        frame[target] = [0.1 * (index + offset) for index in range(studies)]
    return frame


def test_prepared(tmp_path: Path) -> None:
    """BOTH sites are rewritten, and a source spelling it a different number of times is refused."""
    written = _entry(tmp_path).prepared(tmp_path / "patched.py")
    text = written.read_text(encoding="utf-8")
    assert BUDGET_SOURCE not in text
    assert text.count("TIME_BUDGET = 7200.0") == 2, "a single-site patch leaves his rad stage aiming at 8h"
    assert WEIGHTS_SOURCE not in text
    assert "pixel_group" in text, "his weights search must skip a package his loader cannot index"

    once = tmp_path / "once.py"
    once.write_text(f"{BUDGET_SOURCE}\n{WEIGHTS_SOURCE}\n    pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not 2"):
        BendEntry(source=once, working=tmp_path, seconds=HOUR).prepared(tmp_path / "no.py")

    nosearch = tmp_path / "nosearch.py"
    nosearch.write_text(f"{BUDGET_SOURCE}\n{BUDGET_SOURCE}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not 1"):
        BendEntry(source=nosearch, working=tmp_path, seconds=HOUR).prepared(tmp_path / "no.py")


def test_flattened(tmp_path: Path) -> None:
    """Three concatenated cells become one script that compiles; the first future line stays."""
    entry = _entry(tmp_path)
    cells = f"{FUTURE}\nprint(1)\n{FUTURE}\nprint(2)\n{FUTURE}\nprint(3)\n"
    with pytest.raises(SyntaxError):
        compile(cells, "cells.py", "exec")

    one = entry.flattened(cells)
    assert one.count(FUTURE) == 1, "the first one is kept — it is the only legal position for it"
    assert one.count("print(") == 3, "nothing but the duplicate statements is dropped"
    compile(one, "one.py", "exec")


def test_prepared_refuses_a_source_that_does_not_compile(tmp_path: Path) -> None:
    """A cell boundary the flattener does not know about dies HERE, not on a nine-hour slot."""
    broken = tmp_path / "broken.py"
    broken.write_text(f"{BUDGET_SOURCE}\n{WEIGHTS_SOURCE}\n    pass\ndef (:\n{BUDGET_SOURCE}\n", encoding="utf-8")
    with pytest.raises(SyntaxError):
        BendEntry(source=broken, working=tmp_path, seconds=HOUR).prepared(tmp_path / "no.py")


def test_read(tmp_path: Path) -> None:
    """A well-formed table comes back; absent, short-columned and null tables are refusals."""
    entry = _entry(tmp_path)
    with pytest.raises(FileNotFoundError, match="wrote no"):
        entry.read()

    _table().to_csv(tmp_path / SUBMISSION, index=False)
    assert len(entry.read()) == 3

    _table().drop(columns=[TARGETS[0]]).to_csv(tmp_path / SUBMISSION, index=False)
    with pytest.raises(ValueError, match="missing"):
        entry.read()

    holed = _table()
    holed.loc[0, TARGETS[0]] = None
    holed.to_csv(tmp_path / SUBMISSION, index=False)
    with pytest.raises(ValueError, match="1 nulls"):
        entry.read()

    fallback = _table()
    for target in TARGETS:
        fallback[target] = 0.5
    fallback.to_csv(tmp_path / SUBMISSION, index=False)
    with pytest.raises(ValueError, match="CONSTANT"):
        entry.read()
