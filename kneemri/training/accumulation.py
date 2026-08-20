"""Trading time for effective batch size, so the card's width stops choosing the recipe.

It lives beside `Trainer` rather than inside it because `trainer.py` reached the god-file ceiling the
moment this arrived, and of everything in there this is the piece that answers on its own: four total
functions of two integers, no model, no card, no corpus. Its argument is the docstring below.
"""

from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass(frozen=True)
class Accumulation:
    """How many micro-batches share one optimiser step — the encoder's width paid for in time.

    WHAT IT BUYS, and it is the only thing it buys: an EFFECTIVE batch the card cannot hold. Activation
    memory is a property of the micro-batch, so `every=2` at `--batch 6` holds half the activations of
    a step of 12 and updates on the same twelve studies. MEASURED 2026-08-20: dinov2-base at 336 px and
    20 slices asks 56.3 GB at batch 12 against a 32 GB card, so the settled recipe's batch was
    unreachable on the wider encoder and every base arm was confounded by a halved batch it did not
    choose. This is what makes the larger-backbone axis a ONE-VARIABLE test.

    IT IS NOT FREE AND IT IS NOT EXACT. The step costs `every` forward/backward passes, so an epoch
    takes what `every` micro-batches take. And the loss is not identical to the big batch's: anything
    that pools WITHIN a step — the ranking term's pairs, batch statistics — still sees `batch`
    studies, not `batch * every`. Only the gradient is restored.
    """

    every: int = 1
    """One means no accumulation: the loop steps on every micro-batch, which is what every arm in the
    ledger ran under and what the default must therefore stay."""

    def share(self, loss: Float[Tensor, ""]) -> Float[Tensor, ""]:
        """This micro-batch's share of the step's loss.

        The DIVISION is what makes accumulation a bigger batch rather than a bigger learning rate:
        gradients sum across the group, so without it a group of two arrives at the optimiser with
        twice the magnitude and the schedule everything was tuned under is silently doubled.
        """
        return loss / self.every

    def closes(self, micro_steps: int) -> bool:
        """Does this micro-batch complete a group, i.e. is the optimiser stepped now?"""
        return micro_steps % self.every == 0

    def groups(self, micro_batches: int) -> int:
        """How many optimiser steps `micro_batches` micro-batches make — the schedule's step count.

        FLOOR, because the loop drops the trailing partial group. OneCycleLR raises past its total, and
        it raises on the LAST step of the run — after every epoch is logged and before the checkpoint
        is written, the most expensive place to fail. The batcher already drops the partial last BATCH
        for the same class of reason; this is that rule one level up.
        """
        return max(micro_batches // self.every, 1)

    def whole(self, micro_batches: int) -> int:
        """`micro_batches` truncated to complete groups, so no gradient is accumulated and thrown away.

        Without this a trailing group's backward passes cost their full time and then land in the next
        epoch's first step — or, with a zero_grad at the top of the epoch, are discarded outright. The
        cost of dropping them is bounded and known, and it is the same bargain the batcher already
        takes: the order is reshuffled every epoch, so it is a different tail each time.
        """
        return micro_batches - micro_batches % self.every
