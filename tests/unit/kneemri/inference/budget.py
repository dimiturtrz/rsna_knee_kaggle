from types import SimpleNamespace

import pytest

from kneemri.inference import budget as budget_module
from kneemri.inference.budget import KERNEL_SECONDS, Budget

CHUNKS = 18
"""The ~910-study hidden split at 48 studies a chunk, the shape knee-s6n3v was simulated on."""


class _Clock:
    """Wall-clock the test advances by hand, so a nine-hour session runs in no time."""

    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(budget_module, "time", SimpleNamespace(time=fake.time))
    return fake


def _run(budget: Budget, clock: _Clock, seconds_per_chunk: float) -> list[bool]:
    """One KernelRun's chunk sequence on `budget`: whether it was overrunning at each chunk report."""
    budget.begin()
    overrunning = []
    for index in range(CHUNKS):
        budget.chunk(index, CHUNKS)
        overrunning.append(budget.overrunning())
        clock.now += seconds_per_chunk
    return overrunning


def _spent(seconds: float, *, ceiling: float = KERNEL_SECONDS) -> Budget:
    """A budget that has already burned `seconds` against a `ceiling`, without waiting for the clock."""
    budget = Budget(seconds=ceiling, margin=1.0)
    budget.started -= seconds
    return budget


def test_projected():
    """The projection is over the WHOLE split, from the chunks of it that are done.

    Chunks and not read groups: the traversal advances every group together over one pass, so a chunk
    is the same fraction of the run whether the entry serves one read or five, and the decode — the only
    party that knows how long the corpus is — is the one that reports it.
    """
    budget = _spent(3600)
    budget.chunk(1, 2)

    assert budget.projected() == pytest.approx(2 * 3600, rel=0.01), "half the split in an hour is two"

    budget.chunk(3, 4)
    assert budget.projected() == pytest.approx(4800, rel=0.01), "three quarters done projects four thirds"

    assert Budget().projected() == float("inf"), "nothing done projects nothing, rather than dividing by zero"


def test_chunk():
    """Progress is chunks done over chunks there are, and a split always has at least one."""
    budget = _spent(3600)
    budget.chunk(1, 4)
    assert budget.projected() == pytest.approx(4 * 3600, rel=0.01)

    budget.chunk(1, 0)
    assert budget.projected() == pytest.approx(3600, rel=0.01), "an empty split is one chunk, not a zero divide"


def test_measured():
    """No chunk reported means nothing to project from, and the first chunk is not free.

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

    tight.chunk(1, 4)
    assert tight.overrunning()

    roomy = _spent(1.0)
    roomy.chunk(1, 1)
    assert not roomy.overrunning(), "a second against nine hours is not"


def test_report():
    """A measured budget names both hours; an unmeasured one prints nothing rather than infinity."""
    budget = _spent(3600)
    assert budget.report() == "", "no chunk done yet — there is no projection to print"

    budget.chunk(1, 2)
    assert budget.report() == "projecting 2.0h of 9.0h, "


def test_margin():
    """The ceiling a run aims at is under the real one, because the tail is not in the projection."""
    assert Budget(seconds=1000, margin=0.85).seconds == pytest.approx(850)
    assert Budget().seconds < KERNEL_SECONDS, "the last chunk, the rank-combine and the CSV are unmeasured"


def test_begin(clock: _Clock):
    """A second run on a shared budget projects from ITS start, against what the first left of the ceiling.

    knee-s6n3v: `started` was set once, so the second run divided the first run's 2457 s by its own
    first chunks and gave up six of seven members on the hidden split. The ceiling and the two runs'
    costs are the bead's simulation: a soup at 2.7 s a study, then seven members at 3.5 s a study.
    """
    shared = Budget(seconds=5738, margin=1.0)
    soup = 2457 / CHUNKS
    seven = 3185 / CHUNKS

    assert not any(_run(shared, clock, soup)), "the first run fits its ceiling"
    second = _run(shared, clock, seven)
    assert not any(second), "the second run must not be charged the first run's hours"
    assert shared.seconds == pytest.approx(5738 - 2457), "what the first run spent is off the ceiling"

    # The timeout net still holds: a run that genuinely cannot fit is caught on its first measured chunk,
    # and so is a second run that would fit nine hours but not what the first run left of them.
    slow = _run(Budget(seconds=3000, margin=1.0), clock, 200)
    assert not slow[0], "nothing is measured before the first chunk is done"
    assert slow[1], "3600 s of work against 3000 s gives up at the first projection"

    squeezed = Budget(seconds=3000, margin=1.0)
    assert not any(_run(squeezed, clock, soup))
    assert all(_run(squeezed, clock, seven)[1:]), "3185 s of work against the 543 s the first run left"
