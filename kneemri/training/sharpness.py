"""Sharpness-aware minimisation: descend from the WORST point in a ball around the weights, not from the
weights.

It lives beside `Trainer` rather than inside it for the reason `gradient_clip.py` does: `trainer.py` sits
against the god-file ceiling, and this answers on its own -- one float, two calls, no corpus and no card.

WHY THE PROJECT HAS NEVER HAD ONE. Nothing in this tree perturbs the weights before a step; every number
in the ledger was measured descending on the gradient at the current point. The one `SAM` in
`experiments/README.md` is `medsam_dense_s2026`, Segment-Anything used as a TRUNK, an unrelated thing.
The gap this is aimed at is the one the owner keeps naming: the corner sits at roughly 0.865 held out
while the board reads 0.944, and every axis that closed -- aggregation, readout, slice depth, model
scale, corpus, labels, seating, and now weight-space averaging -- closed by moving the offline number
without moving the transfer. SAM is a TRAINING-TIME objective on the flatness of the minimum rather than
a change to what is read, which is a different family from all of them.

THE SCALER NEEDS NO UNSCALING HERE, AND THAT IS THE WHOLE REASON THIS IS ONE CODE PATH. Under fp16 the
gradients in `.grad` are the true gradients times the scaler's factor. The ascent direction is
`rho * g / ||g||`, and a uniform positive factor cancels exactly between the numerator and the norm --
so the perturbation computed from SCALED gradients is the same perturbation, to floating point. Calling
`GradScaler.unscale_` would be not merely unnecessary but WRONG: it may be called only once per optimiser
per step, and `GradientClip.applied` already spends that call immediately before the step. Dodging it is
what lets sharpness-aware minimisation run under the fp16 recipe every dense arm in the ledger used,
against those arms' own numbers, with no bf16 confound and no second control.

WHAT IT IS NOT. It is not weight decay and it is not gradient clipping. Decay bounds where the walk ends
up; clipping bounds how far one update may travel; this changes WHICH DIRECTION the update points, by
asking for the gradient a short distance uphill instead of the gradient here.
"""

import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

FLAT = 0.0
"""No ball at all -- the ordinary step every number in the ledger was measured under, and therefore the
default. A radius of zero is meaningless as a neighbourhood (the perturbed point IS the point), which is
what makes it a safe spelling for "off"."""

FINITE_ONLY = "the gradient norm was not finite"

DECLINE_REPORT = 100
"""How many declines between log lines. The FIRST one is always reported, because an arm in which the
ascent never fires is an ordinary run at half the throughput reporting under the SAM arm's name -- and
nothing else in the tree would say so. After that a count every hundred keeps a pathological run visible
without writing a line per micro-batch."""


class Sharpness:
    """The radius of the ball the step is asked to descend from the worst point of.

    A NULL POLICY OBJECT rather than a branch in `fit_epoch`, like `GradientClip` and `Accumulation`
    beside it: the trainer holds one of these always and calls it always, so the hottest loop this
    project owns grows no `if` and there is no site at which a flag can be silently skipped. The branch
    that remains is INSIDE `climbed`, where it costs one comparison and is the difference between doing
    nothing and walking every parameter twice.

    IT IS NOT FROZEN, unlike its two neighbours, because the ascent it takes has to be undone exactly:
    the offset applied to each parameter is remembered and subtracted back, rather than recomputed from
    a gradient that the second backward pass has by then overwritten.
    """

    def __init__(self, radius: float = FLAT) -> None:
        self.radius = radius
        self.climbs = 0
        self.declines = 0
        self._climb: dict[str, Tensor] = {}

    def climbed(self, model: torch.nn.Module) -> bool:
        """Move the weights to the worst point of the ball, returning whether they were moved.

        The caller owes a matching `descended` on every path where this returns True, and owes a second
        forward and backward pass BETWEEN the two -- the gradient at the perturbed point is the whole
        object of the exercise, and a step taken without it is an ordinary step from the wrong weights.

        THE RETURN VALUE IS A REFUSAL CHANNEL, not a status. Under fp16 an overflowed backward leaves
        `inf` in `.grad`, and the scaler's answer to that is to skip the step and halve its factor. A
        perturbation built from those gradients would be `nan` in every parameter and would not be
        undone by subtracting a `nan` back, so the model would be destroyed by one unlucky batch. On a
        non-finite norm this declines to move at all and the step proceeds as an ordinary one.
        """
        if not self.radius:
            return False
        gradients = {name: parameter.grad for name, parameter in model.named_parameters() if parameter.grad is not None}
        if not gradients:
            return False
        norm = torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(gradient.detach().float()) for gradient in gradients.values()])
        )
        if not bool(torch.isfinite(norm)) or not float(norm):
            self.declines += 1
            if self.declines == 1 or not self.declines % DECLINE_REPORT:
                logger.warning("sharpness declined %d step(s): %s", self.declines, FINITE_ONLY)
            return False
        self.climbs += 1
        if self.climbs == 1:
            logger.info("sharpness climbing at radius %g — the ascent fired", self.radius)
        scale = self.radius / float(norm)
        self._climb = {name: gradient.detach() * scale for name, gradient in gradients.items()}
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self._climb:
                    parameter.add_(self._climb[name])
        return True

    def descended(self, model: torch.nn.Module) -> None:
        """Put the weights back exactly where `climbed` found them, before the optimiser steps.

        SUBTRACTS THE REMEMBERED OFFSET rather than recomputing it. By the time this is called the
        second backward pass has replaced every `.grad`, so a freshly computed `rho * g / ||g||` would
        be a different vector and the restore would leave the model somewhere neither pass visited --
        a drift that compounds once per optimiser step and would read as a bad recipe rather than as a
        bug.
        """
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self._climb:
                    parameter.sub_(self._climb[name])
        self._climb = {}
