"""How much of each study actually reached the model, accumulated across a served split.

THE ONE FAILURE THE SERVE PATH COULD NOT SEE. `Submission.run`'s own docstring legislates that a kernel
which fails must write NO submission, because "a fallback converts a loud failure into a quiet wrong
answer". Routing collapse is that quiet wrong answer arriving without any fallback to blame.
"""

import logging

import numpy as np
from jaxtyping import Float
from torch import Tensor

logger = logging.getLogger(__name__)

ZERO_SLOT_SHARE_MAX = 0.05
"""The share of studies that may reach the model with NO slot at all before the routing contract is
considered broken rather than the data considered hard.

LEGISLATED, not seeded from this tree. A study whose every slot is absent contributes zeros under a zero
mask, so the head has nothing to renormalise over and that row cannot be better than chance whatever the
members are worth. Over all 4,407 corpus studies the measured rate is ZERO — the sparsest study still
fills two of six slots — so any positive rate at serve time is a statement about the serve path, not
about knee MRI. Five percent is the ceiling at which twelve labels each lose about a chance row in
twenty, which is inside the holdout floor and therefore the largest miss worth staying silent about."""

FILL_WITNESS_FLOOR = 100
"""Below this many studies the fill gate observes and reports but never refuses.

The same asymmetry `Submission.check` already legislates for constant columns: the visible test set inside
a kernel is THREE studies, and a rule that fails the run which has to succeed before anything can be
submitted is a rule that costs every slot instead of saving one. A hidden test is thousands and always
clears this."""


class SlotFill:
    """Per-slot coverage of one scheme's decodes, accumulated over a served split.

    TRAINING NEVER ROUTES; SERVING ALWAYS DOES, which is why this asymmetry exists at all. Every training
    module reads a memmap cache that was routed ONCE, offline, over `train_series.csv`, and
    `StudyCache.report` logs per-slot fill and the empty-study count while building it. The kernel routes
    every hidden-test study LIVE through the same predicates and logged nothing. So the one path whose
    coverage we can never inspect afterwards was also the only one not measuring it.

    A series that matches no slot is not an error, it is an absence: the study decodes, the mask reads
    zero, the head renormalises over nothing, the table is well formed, and `check` passes it because
    `check` can only see the table's SHAPE. The score is the only thing that would have said so, and it
    says so a day later and costs a slot.

    IT OBSERVES ALWAYS AND REFUSES RARELY. The per-slot shares go to the log on every run, including the
    three-study visible pass, because the number is worth having even where it cannot be trusted. The
    refusal needs `FILL_WITNESS_FLOOR` studies before it can fire.
    """

    def __init__(self, names: tuple[str, ...]) -> None:
        self.names = names
        self.filled = np.zeros(len(names), dtype=np.int64)
        self.studies = 0
        self.empty = 0

    def observed(self, presence: Float[Tensor, "studies slots"]) -> None:
        """Record one chunk's presence masks, rows are studies and columns are slots."""
        mask = presence.detach().to("cpu").numpy() > 0
        self.filled += mask.sum(axis=0)
        self.studies += mask.shape[0]
        self.empty += int((~mask.any(axis=1)).sum())

    def shares(self) -> dict[str, float]:
        """Per-slot fill share over every study served so far."""
        return {name: float(self.filled[index]) / self.studies for index, name in enumerate(self.names)}

    def report(self) -> None:
        """Log what reached the model, in the same shape `StudyCache.report` logs at build time."""
        if not self.studies:
            logger.warning("slot fill: no studies observed — the decode path never ran")
            return
        for name, share in self.shares().items():
            logger.info(f"slot fill {name}: {share:.3f}")
        logger.info(f"studies with no slot at all: {self.empty}/{self.studies}")

    def refuse_if_collapsed(self) -> None:
        """Raise when too many studies routed to nothing, so the run writes no table instead of a bad one.

        A DECODE FAILURE AND A ROUTING FAILURE ARRIVE AS THE SAME ZEROS and this deliberately does not
        try to separate them. `decoded_at` already logs a traceback per unreadable study, so the two are
        distinguishable in the log; here they are the same thing — a study the model could not see — and
        the response to either at this rate is to fail loudly rather than to score.
        """
        if self.studies < FILL_WITNESS_FLOOR:
            return
        share = self.empty / self.studies
        if share > ZERO_SLOT_SHARE_MAX:
            message = (
                f"{self.empty}/{self.studies} studies ({share:.1%}) reached the model with no slot at all, "
                f"over a ceiling of {ZERO_SLOT_SHARE_MAX:.0%} — the routing contract broke, and these rows "
                f"cannot score better than chance"
            )
            raise ValueError(message)
