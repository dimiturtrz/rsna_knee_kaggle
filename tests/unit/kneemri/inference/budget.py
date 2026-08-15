import pytest

from kneemri.inference.budget import KERNEL_SECONDS, Budget


def _spent(seconds: float, *, ceiling: float = KERNEL_SECONDS) -> Budget:
    """A budget that has already burned `seconds` against a `ceiling`, without waiting for the clock."""
    budget = Budget(seconds=ceiling, margin=1.0)
    budget.started -= seconds
    return budget


def test_projected():
    """The projection is over the WHOLE run — every read group — not the group currently decoding.

    A fused entry decodes once per router, so the chunk loop sees each group's chunks from zero.
    Projecting off that loop alone reads a half-finished run as a finished one, which is how a run that
    was going to overrun looked healthy right up until the session was killed.
    """
    budget = _spent(3600)
    budget.begin(2)
    budget.group(1)

    assert budget.projected() == pytest.approx(2 * 3600, rel=0.01), "half of two groups in an hour is two"

    budget.chunk(1, 2)
    assert budget.projected() == pytest.approx(4800, rel=0.01), "three quarters done projects four thirds"

    assert Budget().projected() == float("inf"), "nothing done projects nothing, rather than dividing by zero"


def test_begin():
    """The number of decodes is the unit of total work, and a run always has at least one."""
    budget = _spent(3600)
    budget.begin(0)
    budget.group(0)
    budget.chunk(1, 1)

    assert budget.groups == 1
    assert budget.projected() == pytest.approx(3600, rel=0.01), "one group, done, is the elapsed time"


def test_group():
    """Entering a group counts the ones before it as whole, and restarts the within-group fraction."""
    budget = _spent(3600)
    budget.begin(4)
    budget.chunk(1, 2)
    budget.group(2)

    assert budget.fraction == 0.0, "a fraction of the previous decode cannot carry into this one"
    assert budget.projected() == pytest.approx(2 * 3600, rel=0.01), "two of four groups done"


def test_chunk():
    """Progress inside a decode moves the projection between whole groups."""
    budget = _spent(3600)
    budget.begin(1)
    budget.chunk(1, 4)

    assert budget.projected() == pytest.approx(4 * 3600, rel=0.01)


def test_measured():
    """No work reported means nothing to project from, and the first chunk is not free.

    It includes the model loads and the CUDA warm-up, so a projection made before any progress is
    reported would fire on every run — giving up members over a cost the run pays once.
    """
    budget = _spent(3600, ceiling=1.0)

    assert not budget.measured()

    budget.chunk(1, 4)
    assert budget.measured()


def test_overrunning():
    """A run is overrunning only once it has measured something AND projects past its ceiling."""
    tight = _spent(3600, ceiling=1.0)
    assert not tight.overrunning(), "an unmeasured run is never overrunning, however small the ceiling"

    tight.begin(1)
    tight.chunk(1, 4)
    assert tight.overrunning()

    roomy = _spent(1.0)
    roomy.begin(1)
    roomy.chunk(1, 1)
    assert not roomy.overrunning(), "a second against nine hours is not"


def test_report():
    """A measured budget names both hours; an unmeasured one prints nothing rather than infinity."""
    budget = _spent(3600)
    assert budget.report() == "", "no chunk done yet — there is no projection to print"

    budget.begin(2)
    budget.group(1)
    assert budget.report() == "projecting 2.0h of 9.0h, "


def test_margin():
    """The ceiling a run aims at is under the real one, because the tail is not in the projection."""
    assert Budget(seconds=1000, margin=0.85).seconds == pytest.approx(850)
    assert Budget().seconds < KERNEL_SECONDS, "the last chunk, the rank-combine and the CSV are unmeasured"
