"""The contrastive term over keys that span steps, rather than keys inside one batch."""

from typing import override

import torch
from jaxtyping import Float
from torch import Tensor

from kneemri.training.grades import Predicted
from kneemri.training.objective import (
    CONTRAST_TEMPERATURE,
    CONTRAST_WEIGHT,
    PAIR_MEMORY,
    PAIR_POOL,
    ContrastiveObjective,
    Shaping,
)
from kneemri.training.pair_memory import PairMemory


class RememberedContrast(ContrastiveObjective):
    """SupCon whose denominator ranges over the remembered steps as well as the live batch.

    THE SUSPECT `contrast_dense` NAMED AGAINST ITSELF. That arm read -0.0087 against its control and the
    ledger pre-registered the harness, not the mechanism: a batch of twelve sees a 7%-prevalence finding
    about once, so for most labels the positives a supervised contrastive term needs are simply absent
    and the anchor drops out forming no term at all. What was measured is therefore "supervised
    contrastive AT BATCH 12", and the named precondition for reading it as anything wider was a
    remembered pool of past readouts. That pool exists here — `PairMemory`, proven by `RememberedRanking`
    — and had never been pointed at this term.

    WIDENING IS THE WHOLE CHANGE. At capacity eight the population a study is contrasted against grows
    from twelve to ninety-six, so a 7% finding is expected in seven of them rather than in one; the
    gradient still flows only through the live step, and the remembered side is a fixed target, exactly
    as in a contrastive memory queue. It is the same trade `RememberedRanking` makes and it is not free:
    a remembered context was produced by weights that have since moved, which is what `PAIR_MEMORY` is
    chosen against.

    IT IS AN INSTRUMENT, NOT AN IMPROVEMENT. The pre-registered reading is in `knee-agrj`: flat inside
    the paired floor closes the lane on price, above it confirms the batch-size artefact and re-opens the
    representation axis, below it closes the lane on mechanism. None of the three is claimed in advance.
    """

    def __init__(
        self,
        weight: float = CONTRAST_WEIGHT,
        shaping: Shaping | None = None,
        *,
        rank_weight: float = 0.0,
        temperature: float = CONTRAST_TEMPERATURE,
        capacity: int = PAIR_MEMORY,
    ) -> None:
        super().__init__(weight, shaping, rank_weight=rank_weight, temperature=temperature)
        self.memory = PairMemory(capacity)

    @override
    def settings(self) -> dict[str, object]:
        """The contrastive account with the pool the memory really holds, asked of the memory itself."""
        return {**super().settings(), PAIR_POOL: self.memory.capacity()}

    @override
    def contrast(self, batch: Predicted) -> Float[Tensor, ""]:
        """SupCon with the live batch as anchors and the live batch plus the remembered steps as keys.

        THE POOL IS BUILT `[live, remembered]` AND NEVER THE REVERSE, which `supcon`'s docstring states as
        its contract: every anchor has to be its own first key so the diagonal of the leading block is the
        only entry the self-exclusion masks out. Appended the other way the mask would fall on a stranger
        and each study would quietly be contrasted against itself.

        The memory is written LAST, as in `RememberedRanking`: remembering before reading would put this
        step's own detached contexts into its own denominator a second time, which is a duplicate key
        rather than a wider population.
        """
        remembered = self.memory.pooled()
        widened = batch if remembered is None else self.widened(batch, remembered)
        loss = self.supcon(batch, widened)
        self.memory.remember(batch)
        return loss

    def widened(self, batch: Predicted, remembered: Predicted) -> Predicted:
        """The live batch followed by the pooled memory, as the one readout `supcon` compares against.

        Only the three fields `supcon` reads are joined honestly; `teacher` is dropped because this term
        never asks for one and a half-carried teacher is worse than none. `weights` is joined even though
        the keys' side is unused, because a `Predicted` whose columns disagree in length is a shape bug
        waiting for the next reader rather than an economy.
        """
        return Predicted(
            logits=torch.cat([batch.logits, remembered.logits]),
            scores=torch.cat([batch.scores, remembered.scores]),
            weights=torch.cat([batch.weights, remembered.weights]),
            embedding=self.joined(batch, remembered),
        )

    def joined(self, batch: Predicted, remembered: Predicted) -> Float[Tensor, "b o h"] | None:
        """Both sides' contexts as one, or None when either side has none — which `supcon` then RAISES on.

        None rather than the live half alone: silently narrowing back to the batch is exactly the
        "trains the plain recipe under the arm's name" failure the guards in `run_setup` exist to
        prevent, and here it would return a plausible number about the term this class was built to widen.
        """
        if batch.embedding is None or remembered.embedding is None:
            return None
        return torch.cat([batch.embedding, remembered.embedding])
