"""The ranking term over pairs that span steps, rather than pairs inside one batch."""

from typing import override

from jaxtyping import Float
from torch import Tensor

from kneemri.training.grades import Predicted
from kneemri.training.objective import (
    PAIR_FLOOR,
    PAIR_MEMORY,
    PAIR_POOL,
    RANK_WEIGHT,
    RankedObjective,
    Shaping,
)
from kneemri.training.pair_memory import PairMemory


class RememberedRanking(RankedObjective):
    """The ranking term, over pairs that span steps rather than pairs inside one batch.

    THE MISMATCH THIS EXISTS FOR. The competition scores one AUC per label over the WHOLE test set —
    a global ordering of every study against every other. `RankedObjective` only ever asks for a local
    one: with a batch of 12 it enforces the ordering of 66 pairs drawn from twelve neighbours, and a
    model can satisfy every batch it is shown while ordering the corpus badly, because nothing in the
    objective compares a study to one it was never batched with. Batch size is a memory decision here —
    twelve studies of six slots of twelve slices already fills a 20 GB budget — so the pair pool is set
    by the card rather than by the statistic.

    Remembering past readouts breaks that link. The pairs grow with the memory, which costs kilobytes,
    instead of with the batch, which costs gigabytes. The gradient still flows only through the current
    step's logits; the remembered side is a fixed target, exactly as in a contrastive memory queue.

    A remembered study is worth slightly less than a live one, since the weights that produced its logit
    have moved. That is a real cost and it is what `PAIR_MEMORY` is chosen against, not a free lunch.
    """

    def __init__(
        self, weight: float = RANK_WEIGHT, shaping: Shaping | None = None, *, capacity: int = PAIR_MEMORY
    ) -> None:
        super().__init__(weight, shaping)
        self.memory = PairMemory(capacity)

    @override
    def settings(self) -> dict[str, object]:
        """The ranked account with the pool the memory really holds, asked of the memory itself."""
        return {**super().settings(), PAIR_POOL: self.memory.capacity()}

    @override
    def ranking(self, batch: Predicted) -> Float[Tensor, ""]:
        """The within-batch pairs and the cross-step ones, pooled as one population.

        Summed before dividing rather than averaged as two ratios, so a pair counts the same whichever
        side of the memory boundary it falls on.

        BOTH DIRECTIONS ARE FORMED. `pairs` is directional, and asking only whether the live batch
        outranks the remembered one would silently drop every pair a remembered study should win — half
        the comparisons, and the half biased toward whichever readings happen to be strong right now.
        """
        misordered, pair_weight = self.pairs(batch, batch)
        remembered = self.memory.pooled()
        if remembered is not None:
            for above, below in ((batch, remembered), (remembered, batch)):
                across, mass = self.pairs(above, below)
                misordered = misordered + across
                pair_weight = pair_weight + mass
        self.memory.remember(batch)
        return misordered / pair_weight.clamp(min=PAIR_FLOOR)
