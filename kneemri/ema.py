"""An exponentially-moved average of the weights, kept alongside the ones being trained.

Every strong public notebook for this competition keeps one, and the reason fits what we measured:
three seeds of an identical recipe spread 0.0062, which is the same run-to-run wobble an average over
recent steps is meant to damp. An EMA is not a better model so much as a less arbitrary one — it stops
the answer depending on exactly which step training happened to stop at.

Kept as its own object rather than folded into the trainer so it can be tested without a training loop,
and so a run that does not want one pays nothing.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from torch import Tensor, nn

DECAY = 0.999
"""How much of the average survives one step.

At 0.999 the average has a horizon of roughly 1/(1-decay) = 1000 steps, which is about three epochs at
our 302 steps per epoch — long enough to smooth a run's wobble, short enough to follow a one-cycle
schedule down to its final learning rate rather than lagging a stale plateau.
"""

WARMUP_FLOOR = 10
"""Steps over which the decay is ramped in.

Without it the average spends its first thousand steps dominated by the random initialisation, which is
the one set of weights we know to be worthless.
"""


class WeightAverage:
    """A shadow copy of the model's floating-point parameters, moved toward them each step."""

    def __init__(self, model: nn.Module, decay: float = DECAY) -> None:
        self.decay = decay
        self.steps = 0
        self.shadow: dict[str, Tensor] = {
            name: value.detach().clone().float()
            for name, value in model.state_dict().items()
            if value.is_floating_point()
        }

    def update(self, model: nn.Module) -> None:
        """Move the average toward the live weights by one step.

        The effective decay is ramped by step count, so early steps average over what little history
        exists instead of over the initialisation.
        """
        self.steps += 1
        rate = min(self.decay, (1.0 + self.steps) / (WARMUP_FLOOR + self.steps))
        state = model.state_dict()
        for name, value in self.shadow.items():
            value.mul_(rate).add_(state[name].detach().float(), alpha=1.0 - rate)

    def state_dict(self, model: nn.Module) -> dict[str, Tensor]:
        """The averaged weights, in the model's own dtypes, with non-float entries carried through.

        Integer buffers — counters, index tables — are not averaged and must be taken as they are; a
        state dict missing them does not load.
        """
        live = model.state_dict()
        return {
            name: (self.shadow[name].to(value.dtype) if name in self.shadow else value.detach().clone())
            for name, value in live.items()
        }

    @contextmanager
    def applied(self, model: nn.Module) -> Iterator[None]:
        """Run a block with the averaged weights in place, then put the live ones back.

        Scoring an epoch has to happen under the weights that would be SAVED, or the number chooses an
        epoch for a model that is not the one written to disk.
        """
        live = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model.load_state_dict(self.state_dict(model))
        try:
            yield
        finally:
            model.load_state_dict(live)
