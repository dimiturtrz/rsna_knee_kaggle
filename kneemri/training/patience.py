"""Stopping a run once the holdout has stopped improving, without shortening its schedule.

MEASURED 2026-08-13 across every arm's metrics file: the twelve-epoch OneCycle peaks at epoch 6 and
then does nothing for five more — 0.8457, then 0.8410, 0.8394, 0.8395, 0.8391, 0.8394. Every DINOv2
whole-stack arm has this shape (wsfold0, res392, sliceattn, slices16, extended_ws, fold1 all peak at
6 of 12). Because the run keeps its BEST epoch's weights, those five epochs produce checkpoints that
are thrown away: roughly nineteen minutes an arm, about 42% of its wall-clock.

THAT 42% IS THE WASTE, NOT THE PRIZE, and the difference is most of it. Measured 2026-08-20 over all
55 metrics files by `kneemri.experiments.stale_gap`: the longest non-improving stall any run RECOVERED
from is 3 epochs (`lora_r8_v3small_224`), so the smallest value that never truncates a run that was
about to improve is 4 — and a stopper has to sit through its own patience before it fires. At 4, the
corpus gives back 42 of its 163 discarded epochs, **26% of the waste**, because a twelve-epoch arm
that peaks at 6 wastes five and four of those five are spent waiting. The saving is concentrated in
the LONG runs, which are exactly the arms asking a question about the tail, where stopping deletes the
measurement. Real, small, and pointed at the arms that can least use it.

AND `band_dense_long` IS THE CASE THAT PRICES THE TRADE INSTEAD OF ASSERTING IT. It is now the corpus's
largest single waster — 24 epochs, peak at 6, **17 epochs run past it, about 103 minutes of card** — and
it is also a shape arm, so it is precisely the run the paragraph above says a stopper would ruin. Work
the counterfactual on its actual curve rather than on the category. After the peak (0.8587) it scores
0.8479, 0.8554, 0.8581, 0.8521: four non-improving epochs, so patience 4 fires at epoch 10 and saves 13
of the 17 — a third of the whole corpus saving from one run, and 79 minutes of card.

WHAT THAT WOULD HAVE COST IS THE STRENGTH OF THE VERDICT, NOT THE VERDICT. Stopped at epoch 10 the arm
still reads a decline of 0.0066 from its peak, which clears the 0.0062 floor — the overfitting call is
reachable, at 1.06x the floor. Run to 24 it reads 0.0182, 2.9x the floor, which is the difference between
a number that survives a sceptic and one that does not. So the honest rule is not "shape arms never stop"
but: a stopper trades card time for CONFIDENCE in the tail, at roughly 6 minutes an epoch, and the arms
worth paying that for are the ones whose conclusion is about the tail. `band_dense_long` was one and the
79 minutes were well spent; a rate or pooling arm that peaks at 6 and is read on its peak is not.

THIS IS NOT "RUN FEWER EPOCHS", which is a different and worse change. OneCycle is defined over
`total_steps`, so asking for eight epochs COMPRESSES the whole schedule rather than truncating it —
measured head to head the eight-epoch cycle leads early (0.7510 against 0.7393 at epoch 0) and trails
at the end (0.8429 against 0.8457). The slow anneal earns its keep. The schedule is therefore still
built for the full epoch count and only the LOOP ends early.

NOR A FIXED STOP, and the exceptions are what carry the mechanism: intensity392 peaks at epoch 9,
DINOv3-small at 11 of 15, DINOv3 at 224 px at 8. Heavier augmentation and an unfamiliar encoder both
converge later, which is what a harder fit should do — so a blanket "stop at 7" would have cost
exactly the arms that needed the time. A counter of epochs since the last improvement adapts to that
where a constant cannot.

IT CANNOT CHANGE A RECORDED BEST. The best epoch is already the one whose weights are kept, so
stopping after it only removes epochs that would have been discarded.
"""


class Patience:
    """How many epochs a run may go without improving before it stops."""

    def __init__(self, epochs: int = 0) -> None:
        self.epochs = epochs
        self.stale = 0

    def exhausted(self, *, improved: bool) -> bool:
        """Whether the run should stop now, counting the epochs since the holdout last improved.

        `epochs=0` disables stopping entirely, which is the default so that no arm already in the
        ledger changes meaning by being re-run.

        A run with no holdout (`--fold -1`) ranks its epochs by epoch NUMBER, which increases every
        time, so `improved` is always true and this never fires — the final fit runs its full schedule
        by construction rather than by a special case.
        """
        if improved:
            self.stale = 0
            return False
        self.stale += 1
        return self.epochs > 0 and self.stale >= self.epochs
