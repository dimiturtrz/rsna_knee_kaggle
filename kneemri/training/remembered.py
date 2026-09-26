"""Any pooled term, read over pairs that span steps rather than pairs inside one batch."""

from collections.abc import Callable
from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn

from kneemri.training.grades import Predicted
from kneemri.training.pair_memory import PairMemory
from kneemri.training.terms import PAIR_MEMORY, PAIR_POOL, Pooled, Term


class Remembered(Term):
    """A `Pooled` term widened by a remembered pool of past readouts.

    THE MISMATCH THIS EXISTS FOR. The competition scores one AUC per label over the WHOLE test set — a
    global ordering of every study against every other. A batch of 12 enforces 66 pairs drawn from twelve
    neighbours, and a finding positive in ~7% of studies is absent from roughly half of all batches, so
    the margin's class means and the contrast's positives are simply missing for the rarest labels. Batch
    size is a memory decision here — twelve studies already fill a 20 GB budget — so the population is
    set by the card rather than by the statistic.

    Remembering past readouts breaks that link. The population grows with the memory, which costs
    kilobytes, instead of with the batch, which costs gigabytes. The gradient still flows only through the
    current step; the remembered side is a fixed target, exactly as in a contrastive memory queue. A
    remembered study is worth slightly less than a live one, since the weights that produced it have moved
    — the cost `PAIR_MEMORY` is chosen against.

    One wrapper for every pooled term, because the three copies it replaces differed only in which
    population the wrapped term ranged over — which is what `Pooled.across` names.
    """

    def __init__(self, term: Pooled, capacity: int = PAIR_MEMORY) -> None:
        super().__init__(term.weight)
        self.term = term
        self.memory = PairMemory(capacity)

    def __call__(self, batch: Predicted) -> Float[Tensor, ""]:
        """The wrapped term over the live batch and the remembered steps, then this batch remembered.

        The memory is written LAST: remembering before reading would put this step's own detached
        readouts into its own population a second time, which is a duplicate rather than a wider pool.
        """
        remembered = self.memory.pooled()
        loss = self.term(batch) if remembered is None else self.term.across(batch, remembered)
        self.memory.remember(batch)
        return loss

    @override
    def graded(self, batch: Predicted, spread: Callable[[Predicted], Predicted]) -> Float[Tensor, ""]:
        """The wrapped term as a BASE, over the live batch widened by the memory and only THEN spread.

        Widening before the grade policy restates the batch keeps the memory holding the twelve latent
        readings rather than one policy's threshold crossings, so a pool outlives nothing but the step.
        """
        remembered = self.memory.pooled()
        population = batch if remembered is None else self.term.widened(batch, remembered)
        loss = self.term.graded(population, spread)
        self.memory.remember(batch)
        return loss

    @override
    def settings(self) -> dict[str, object]:
        """The wrapped term's account with the pool the memory really holds, asked of the memory itself."""
        return {**self.term.settings(), PAIR_POOL: self.memory.capacity()}

    @override
    def attached(self, device: torch.device) -> list[nn.Parameter]:
        """Whatever the wrapped term owns — the memory itself holds only detached readouts."""
        return self.term.attached(device)
