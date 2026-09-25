"""One raptor seat beside our named members: the whole of a raptor-first kernel's `main()`.

Forty-three kernels carried this sequence as a copied body, differing only in the members they name, the
foreign share and the vote. It is written once here so an entry is its member choice and nothing else;
the kernel script keeps `chosen = (...)` literal because `champion_check` and `submit_check` read the
selection statically.
"""

import logging
import time
from dataclasses import dataclass, field

import pandas as pd
import torch

from core.mounts import Mounts
from kneemri.device import VisibleCards
from kneemri.evaluation.raptor_model import RaptorMember
from kneemri.inference.budget import KERNEL_SECONDS, Budget
from kneemri.inference.ensemble import Serving
from kneemri.inference.vote import LogitAverage, Vote
from kneemri.kaggle.kernel_run import KernelRun
from kneemri.kaggle.raptor_run import DEFAULT_WEIGHTS, WORKING, RaptorEntry
from kneemri.kaggle.raptor_seat import SEATS, RaptorSeat

logger = logging.getLogger(__name__)

SUBMISSION = "submission.csv"
SCRATCH = "raptor"


@dataclass(frozen=True)
class SeatedEntry:
    """Our named members served under one vote, with the raptor member blended in as one weighted seat."""

    names: tuple[str, ...]
    vote: Vote = field(default_factory=LogitAverage)
    weight: float = 1.0 / SEATS
    weights: str = DEFAULT_WEIGHTS
    overlap: bool = False

    def run(self) -> None:
        """Score the raptor member, then ours on the session's remaining budget, then write the blend.

        The raptor member runs FIRST and into a scratch directory, because both producers write a file
        called `submission.csv` and the one that runs last is the one Kaggle would otherwise score.
        """
        scratch = WORKING / SCRATCH
        scratch.mkdir(parents=True, exist_ok=True)
        started = time.time()
        mounts = Mounts()
        device = str(VisibleCards().device())
        member = RaptorMember.loaded(RaptorEntry.weights(self.weights, mounts), device)
        theirs = RaptorEntry(mounts.one("sample_submission.csv", "competition data"), working=scratch).run(
            member, device
        )
        # Ours load onto the same T4. Dropping the reference is not enough on its own: torch keeps the
        # freed blocks in its caching allocator, and a model's worth of them is the margin an OOM eats.
        del member
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        KernelRun(
            names=self.names, overlap=self.overlap, serving=Serving(vote=self.vote, budget=self._left(started))
        ).run()
        ours = pd.read_csv(WORKING / SUBMISSION)
        RaptorSeat(weight=self.weight).blended(ours, theirs).to_csv(WORKING / SUBMISSION, index=False)

    def _left(self, started: float) -> Budget:
        """The SESSION's budget less what the raptor pass cost, so both producers do not aim at one ceiling.

        Without it the kernel is killed at the real ceiling with no submission, a budget defeated by its
        own assumption that it starts at zero.
        """
        spent = time.time() - started
        logger.info("the raptor pass cost %.0fs; the %d of ours get what is left", spent, len(self.names))
        return Budget(seconds=max(KERNEL_SECONDS - spent, 0.0))
