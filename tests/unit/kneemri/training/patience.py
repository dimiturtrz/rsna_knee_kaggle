import pytest

from kneemri.training.patience import Patience

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


RAMP_END = 0.25
EPOCHS = 36


def test_settled():
    """Classes: off settles all, a ramp splits at its end, near-one keeps the last, out of range raises."""
    assert all(Patience().settled(epoch, EPOCHS) for epoch in range(EPOCHS)), "0.0 is today's behaviour"

    ramped = Patience(PATIENCE, RAMP_END)
    assert not ramped.settled(0, EPOCHS)
    assert not ramped.settled(8, EPOCHS), "epoch 8 still trains inside the ramp, which ends at 9/36"
    assert ramped.settled(9, EPOCHS), "the first epoch starting at the ramp end is settled"

    late = Patience(PATIENCE, 0.99)
    assert not late.settled(EPOCHS - 2, EPOCHS)
    assert late.settled(EPOCHS - 1, EPOCHS), "the last epoch always settles, so a run keeps something"

    for outside in (-0.1, 1.0):
        with pytest.raises(ValueError, match="no epoch"):
            Patience(PATIENCE, outside)
