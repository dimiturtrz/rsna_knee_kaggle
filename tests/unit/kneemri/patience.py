from kneemri.patience import Patience

PATIENCE = 3


def test_exhausted():
    """The three equivalence classes: disabled, improving, and stale for long enough to stop.

    The reset is the part worth pinning. A counter that only ever increments would stop a run that
    improves every other epoch, which is the shape `intensity392` (peak at 9) and `dinov3_v3small`
    (peak at 11 of 15) actually have — those arms must run on, and a blanket epoch cap is precisely
    what this exists instead of.
    """
    disabled = Patience()
    assert not any(disabled.exhausted(improved=False) for _ in range(20)), "0 epochs means never stop"

    patience = Patience(PATIENCE)
    assert not patience.exhausted(improved=True)
    assert not patience.exhausted(improved=False)
    assert not patience.exhausted(improved=False)
    assert patience.exhausted(improved=False), "three epochs without improvement is the whole budget"


def test_exhausted_resets_on_an_improvement():
    """A late improvement buys the run its full budget again, rather than a remainder."""
    patience = Patience(PATIENCE)
    patience.exhausted(improved=False)
    patience.exhausted(improved=False)

    assert not patience.exhausted(improved=True), "improving must not stop the run"
    assert patience.stale == 0
    assert not patience.exhausted(improved=False)
    assert not patience.exhausted(improved=False)
    assert patience.exhausted(improved=False)
