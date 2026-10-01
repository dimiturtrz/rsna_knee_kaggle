"""The pre-registration chain: the real arm tree becomes a report that `reading_check` can actually emit.

A CHECK WITH NO TEST THAT IT RUNS IS A CHECK THAT CAN DIE SILENTLY, and this one did. From 2026-09-07
`84660bf5` until 2026-09-08 `813c6d56`, every invocation of `reading_check` exited 1 on
`NotImplementedError: Non-relative patterns are unsupported` — `pathlib.Path.glob` refuses an absolute
pattern, and five arms declare their ruler as an absolute
`D:/data/volumetric/mri/models/...holdout.npz`. It reported nothing at all for a day, and every gate
stayed green throughout, because nothing in `noxfile.py` invokes it.

The unit mirror could not have caught that. `devtools.hygiene.small` forbids a unit test from reading an
external data root, so every case there runs against a `tmp_path` the test itself built — which is exactly
the shape that never contains an absolute declaration written by a real author. The defect lived in the gap
between a checker that passes on synthetic input and an arm tree that is not synthetic.

So the claim here is deliberately weak and is the only one worth making: against the REAL arms directory,
`reading_check` PRODUCES A REPORT rather than raising. Not that the tree is clean — the tree's cleanliness
is the thing the check is for, and asserting it here would make an unrelated new arm fail this test. Exit
0 or exit 1 are both correct outcomes; a traceback is not.
"""

from pathlib import Path

import pytest

from core.paths import DataRoot
from kneemri.experiments.reading_check import ARMS, ReadingDrift


def test_report_over_the_real_arm_tree() -> None:
    """Every arm on disk is surveyable: the report is emitted, whatever verdict it carries."""
    if not ARMS.is_dir():
        pytest.skip("the arm tree is not checked out here")

    root = DataRoot.from_config(Path("paths.yaml"))
    drift = ReadingDrift(root.models(), root.processed(), ARMS)

    lines = drift.report()

    assert lines, "the survey line is emitted unconditionally, so an empty report means the walk died"
    assert "arms:" in lines[-1]


def test_absolute_declarations_resolve_over_the_real_tree() -> None:
    """The exact shape that killed the check: an absolute glob is answered, not raised on.

    `found` is called once per declared reading, so a single arm spelling its ruler absolutely took the
    whole report down. Reaching every declaration on the real tree is what makes this a regression test
    rather than a restatement of the unit case.
    """
    if not ARMS.is_dir():
        pytest.skip("the arm tree is not checked out here")

    root = DataRoot.from_config(Path("paths.yaml"))
    drift = ReadingDrift(root.models(), root.processed(), ARMS)

    declared = [glob for arm in drift.declarations() for glob in arm.readings()]

    assert declared, "arms declare readings on this tree; an empty list means the parse stopped finding them"
    assert all(isinstance(drift.found(glob), bool) for glob in declared)
