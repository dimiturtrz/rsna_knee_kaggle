"""The array a training run already wrote for its own holdout, offered in place of an inference pass.

Its own module because it is its own subject: `holdout_predictions` answers what a member says about
these studies, `prediction_cache` answers whether anyone already asked, and this answers whether the
question was already ANSWERED, by the run itself, before anyone thought to ask it here.
"""

import numpy as np
from jaxtyping import Float

from kneemri.inference.holdout_slab import HoldoutSlab
from kneemri.inference.members import RESIZED_FROM, Member, Window
from kneemri.inference.test_time import SpatialViews
from kneemri.runs.holdout_record import FOREIGN, HoldoutRecord


class TrainerRecord:
    """A member's own `.holdout.npz`, offered in place of an inference pass when it is the SAME reading.

    The trainer already ran the whole holdout through the kept epoch and wrote the result beside the
    checkpoint, exactly. Recomputing it costs the card fifteen minutes a member — measured on
    `cnx_dense_s2026` — which is what made a 95-member sweep a twenty-four hour job against a two hour
    budget, and why the sweep that motivated this was killed rather than finished.

    THE SUBSTITUTION IS AN EQUALITY, NOT AN APPROXIMATION, AND IT WAS CHECKED. `member_curve` over the
    ten members that DID have a cache entry reproduces `record_mix`'s record-read alone-scores to the
    four decimals both print, ten of ten. What was never checked is equality under some OTHER reading —
    a wider window, a rotation average — and the trainer did not compute one, so every guard below is
    about refusing exactly that. A refusal costs the old behaviour; a wrong acceptance silently scores
    a member as a function it was never read as, which is how knee-y8y reached the leaderboard.
    """

    def __init__(self, views: SpatialViews) -> None:
        self.views = views

    def instead_of(
        self, member: Member, slab: HoldoutSlab, window: Window, studies: list[str]
    ) -> Float[np.ndarray, "studies targets"] | None:
        """The recorded probabilities in the caller's study order, or None when the record cannot stand in.

        None for any of six reasons, each of which makes the record a different function from the one
        asked for: the run left no record; it is FOREIGN, so the array came through someone else's
        decode and the feed alone is worth -0.0348; the caller averages over rotations the trainer did
        not; the member is being READ at a size it was not fitted at; the caller reads a window the
        member is not fitted under (`--overlap`, above all); or the record does not cover every study
        asked about, which a fold or corpus rebuild causes.

        DERIVED records ARE accepted. A back-filled array is our own weights read a second time and
        differs only by precision, which is the same tolerance an uncached recomputation here would
        carry — refusing it would send the caller to the card for an array of equal standing.
        """
        stored = HoldoutRecord(member.file).read()
        if stored is None or stored.source == FOREIGN or not self.views.identity:
            return None
        # A RESIZED MEMBER IS A DIFFERENT READING AND THE TRAINER NEVER TOOK IT. The record was written
        # at the fitted size; handing it back for a member read at another one prices a resolution by
        # returning the number the resolution was supposed to move. Measured 2026-09-06: six members
        # "re-derived at 224 px" in under a minute, all six records byte-identical to their native ones.
        # `train_size` differing from `img` is a FITTED choice whose record answers for it, so the test
        # is the mark `resized` leaves and never the sizes themselves.
        if RESIZED_FROM in member.config:
            return None
        if window != member.window(slab.corpus.slices, overlap=False):
            return None
        rows = {name: position for position, name in enumerate(stored.studies)}
        if any(name not in rows for name in studies):
            return None
        return stored.predictions[np.array([rows[name] for name in studies], dtype=np.int64)]
