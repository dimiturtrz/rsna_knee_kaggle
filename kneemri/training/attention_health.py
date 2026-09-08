"""Is the encoder still ROUTING, or has the objective destroyed its attention?

WHY THIS EXISTS. `pretrain_unitfix` drove its reconstruction loss 0.9122 -> 0.5570 over eight epochs,
which is an objective fitting, and it was read that way for a fortnight. Researcher D251/D252 then
measured five checkpoints at n=240 and found FOUR OF FOUR adapted encoders degenerate at the top, while
`adapt_null` (`--lr 0`) reproduced stock to three decimals on every block.

THE COLLAPSE HAS TWO OPPOSITE SHAPES, which is the whole design of this gate and the correction that
retired its first version. `adapt_small_e8_unitfix` fails TOWARD uniform: blocks 8-11 at 5.372 / 5.507 /
5.532 / 5.512 against `ln(257) = 5.549`. The other three fail AWAY from it: `adapt_small_e6` block 11 at
0.000 nats with every patch query sending >99% of its mass to the class token, `adapt_small_e6_lr1e5` at
0.035 with ~99% on ONE PATCH key, `pretrain_small_matched_e8` at 0.023. A margin-to-`ln(N)` test — which
is what this module shipped first — reads those three as HEALTHIER THAN STOCK and passes the worst
checkpoint on disk most comfortably of all.

So the reference is STOCK'S OWN PER-BLOCK PROFILE and the test is two-sided. It is a RATIO and not a
number of nats, because nats do not survive a change of configuration. An absolute margin was tried
first, sized at 0.8 off D252's separation, and it refused a healthy encoder within three optimiser steps
at 5e-5 — exactly the failure researcher D253 predicted for it, since the configuration it fired in
spans 2.48 nats against the 1.7 its margin was fitted in. What IS scale-free is how much of stock's own
distance to each end a block still holds: see `RETENTION_FLOOR`.

The reference is MEASURED AT EPOCH ZERO rather than pasted from the letter. The trainer starts from the
stock checkpoint, so the honest baseline is free, and a constant array would be wrong the moment the
image size, the patch, the trunk or the cache changed — which is the same defect the first version's
`ln(N)` had one layer down.

THE STATISTIC IS THE RAW SOFTMAX ROW OVER ALL KEYS, nothing dropped and nothing renormalised. D251
dropped the class column and renormalised, and on a block where a patch query sends nearly all its mass
to the class token that divides ~0 by ~0 — it reported `entropy 0.000, max_weight 0.000`, and a max
weight of zero is impossible for a distribution. Every number here is a property of the model.

`prefix_mass` is logged beside the entropy because it NAMES which degeneracy arrived: 1.000 is an
attention sink on the class token, 0.000 at low entropy is a single-patch lock. It is one line off the
same tensor and it costs nothing.

THE DRIFT GATE CANNOT SEE ANY OF THIS, in writing: `encoder_adapt_gentle.running.sh:17-19` pre-registers
`adapt_small_e6_lr1e5` at b10 0.1216 / b11 0.2152 and concludes the adaptation "still HAPPENED ... it
just spent itself where a representation can afford to move". That block 11 now measures ~99% of its
attention on one patch. A small, top-concentrated weight move is enough to destroy routing.

It is a REFUSAL and not a log line for the same reason the first version was: a number printed in a log
that nobody was looking for is how the first eight card-hours were spent.
"""

import logging
import math
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from types import TracebackType
from typing import cast

import torch
from jaxtyping import Float
from torch import Tensor, nn

logger = logging.getLogger(__name__)

EAGER = "eager"


class _Readable:
    """Put a model into the state a health reading requires, and put it back exactly as it was.

    TWO things, because the first real run of this gate was defeated by each of them in turn.

    THE KERNEL. Transformers 5 serves `sdpa`, whose fused kernel never materialises the attention
    matrix, so `output_attentions=True` returns a tuple of `None` silently and the gate died on its own
    refusal rather than reading stock.

    THE MODE. A model in `train()` applies stochastic depth and dropout, which are drawn FRESH on every
    forward — so two readings of one encoder would differ by their random draws and not by their weights,
    and the gate subtracts exactly that difference. This was fixed on the argument, not on a measurement:
    it was suspected of causing an early refusal and did not cause it. The `--lr 0` control settled the
    question — with the weights pinned, every block of both epochs reads +0.00 against the reference, so
    the reading is now deterministic to two decimals and any deviation it reports IS the weights. Both
    states are restored on exit because the fused kernel and train mode are what the run trains under.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.previous = EAGER
        self.training = False

    def _switch(self, implementation: str) -> None:
        """`nn.Module.__getattr__` types every unknown attribute as a Tensor, so the call needs a name."""
        switch = cast(Callable[[str], None], self.model.set_attn_implementation)
        switch(implementation)

    def __enter__(self) -> None:
        self.previous = str(getattr(self.model.config, "_attn_implementation", EAGER))
        self.training = self.model.training
        self._switch(EAGER)
        self.model.eval()

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self._switch(self.previous)
        self.model.train(self.training)


RETENTION_FLOOR = 0.25
"""How much of stock's own distance to an end a block must still hold, as a FRACTION, on both sides.

Toward uniform the quantity is the headroom `ln(keys) - H`, and what is required is that a block keep at
least this share of the headroom stock had. Away from uniform it is `H` itself against stock's `H`. Both
are ratios to the reference, so neither carries a unit and neither is tied to a key count.

Sized off D252's five checkpoints, in the units the gate actually uses. Against `ln(257) = 5.549`:

    unitfix b8-11 keep  11.4% /  2.7% / 0.7% /  2.9% of stock's headroom (5.372, 5.507, 5.532, 5.512)
    e6 / e6_lr1e5 / matched_e8 b11 keep 0.0% / 0.8% / 0.5% of stock's entropy (0.000, 0.035, 0.023)
    stock keeps 100% of both, on every block, by construction

So the two populations are separated by the whole band from 11.4% to 100%, and 0.25 sits inside it with
room on both sides. This is what the absolute margin could not do: 0.8 nats was fitted where stock spanned
1.7 nats, and the first configuration it met spanned 2.48, where it refused an encoder that on this ruler
keeps 71-89% of stock on every block. A fraction of the reference travels; a number of nats does not."""


@dataclass(frozen=True)
class AttentionProfile:
    """One reading of an encoder's attention: entropy per block, and where the prefix mass sits.

    The two travel together because either alone is ambiguous. An entropy of 0.03 is degenerate whether
    the mass went to the class token or to one patch, and only `prefix_mass` says which — that is the
    difference between an attention sink and a single-patch lock, and they are different bugs.
    """

    entropy: tuple[float, ...]
    prefix_mass: tuple[float, ...]
    keys: int = 0
    """How wide a softmax row was, patches plus prefix. Carried on the reading rather than declared,
    because the prefix is the trunk's business — one class token for DINOv2, four registers more for
    DINOv3 — and `ln(keys)` is what a uniform row would have scored."""

    class_mass: tuple[float, ...] = ()
    """The share of `prefix_mass` sitting on the CLASS TOKEN alone, which is the half that is a defect.

    `prefix_mass` sums the whole prefix, and at DINOv2's prefix of one that is the same number. At a
    trunk with registers it is not, and the difference is the entire claim of Darcet et al.: a sink
    parked on the REGISTERS is the mechanism working as designed, and a sink on the class token is the
    pathology registers exist to prevent. Summed, the two are indistinguishable.

    Measured 2026-09-08 on `adapt_latent_v3small_m30_e8`, which cleared the gate at mask 0.30 where
    dinov2-small refused, and then printed a LARGER prefix mass than the trunk it was meant to fix
    (b11 0.83 against 0.75). Whether that reads as a success or a worse failure is exactly the split
    this field carries, and the arm was queued to answer it with a statistic that could not.

    It sits BEHIND `keys` rather than beside `prefix_mass`, where it reads better, because callers
    build this dataclass positionally — a third field would have bound a key count into a mass and
    every reading would have stayed plausible.
    """

    def register_mass(self) -> tuple[float, ...]:
        """Per block, the prefix mass NOT on the class token — the share a register trunk parks benignly.

        Subtraction rather than a third measurement, because the two measured quantities are nested: the
        class token is the first prefix key, so their difference is exactly the registers and cannot
        disagree with them. On a trunk with no registers this is zero on every block by construction,
        which is the correct reading and not a missing one.
        """
        return tuple(whole - single for whole, single in zip(self.prefix_mass, self.class_mass, strict=True))

    def report(self) -> str:
        """Both statistics per block, so a log line is a reading and not a single number."""
        entropies = "  ".join(f"{value:.2f}" for value in self.entropy)
        masses = "  ".join(f"{value:.2f}" for value in self.prefix_mass)
        line = f"H {entropies} | prefix {masses}"
        if not self.class_mass:
            return line
        classes = "  ".join(f"{value:.2f}" for value in self.class_mass)
        registers = "  ".join(f"{value:.2f}" for value in self.register_mass())
        return f"{line} | cls {classes} | registers {registers}"


@dataclass(frozen=True)
class AttentionHealth:
    """The stock profile a run is measured against, and the deviation it is refused outside of."""

    tokens: int
    reference: AttentionProfile
    probe: Float[Tensor, "b c h w"] | None = None
    """The EXACT images the reference was read on, held so every later reading is read on them too.

    Held rather than redrawn because the gate subtracts two readings and calls the difference damage, and
    a fresh batch puts image-to-image variation into that subtraction. A gate whose null hypothesis is a
    different sample is not a gate. It was suspected of an early refusal and was not its cause — freezing
    it changed the numbers by less than 0.02 — but the `--lr 0` control that later settled the question
    only reads +0.00 on every block BECAUSE the batch is held, so it is what makes the reading falsifiable."""

    floor: float = RETENTION_FLOOR

    @classmethod
    def baseline(cls, model: nn.Module, images: Float[Tensor, "b c h w"], tokens: int) -> "AttentionHealth":
        """Measure the reference off the model AS IT STARTS, which is the stock checkpoint.

        Measured rather than pasted: a constant array is wrong the moment the image size, the patch, the
        trunk or the cache changes, and it is exactly one forward pass to be right instead. The images
        are kept, so what follows is a comparison of two encoders and not of two batches.
        """
        held = images.detach().clone()
        health = cls(tokens=tokens, reference=AttentionProfile((), ()), probe=held)
        return cls(tokens=tokens, reference=health.profile(model), probe=held)

    def span(self) -> float:
        """How wide the stock profile itself is — logged because it is what broke the previous ruler.

        The absolute margin was sized where this read 1.7 nats and fired where it reads 2.48. It no longer
        decides anything, since a retained FRACTION of stock is the same test at any span, but a run still
        prints it: it is the one number that says how far the configuration sits from D252's, and the next
        surprise on this axis will be read against it.
        """
        if not self.reference.entropy:
            return 0.0
        return max(self.reference.entropy) - min(self.reference.entropy)

    def retention(self, profile: AttentionProfile) -> tuple[float, ...]:
        """Per block, the SMALLER of the two shares of stock the reading still holds.

        Two sides, one number, because a block is degenerate if it has run out of room at EITHER end and
        the gate should report whichever end it is closer to losing. Toward uniform that share is the
        headroom `ln(keys) - H` over stock's headroom; away from it, `H` over stock's `H`. Stock scores
        1.0 on both against itself; every degenerate cell D252 measured scores below 0.115 on one of them.
        """
        uniform = self.uniform()
        blocks = zip(profile.entropy, self.reference.entropy, strict=True)
        return tuple(
            min(
                self._share(uniform - value, uniform - stock),
                self._share(value, stock),
            )
            for value, stock in blocks
        )

    def _share(self, value: float, stock: float) -> float:
        """`value / stock`, with a reference of zero meaning stock had no room to lose and cannot lose it."""
        if stock <= 0.0:
            return 1.0
        return max(0.0, value) / stock

    def uniform(self) -> float:
        """The entropy of attention spread evenly over every key — one end of the two-sided failure.

        Reported rather than used as the threshold: it is what `adapt_small_e8_unitfix` walked into, and a
        log line saying how far a block sits from it is how that walk is read. It is NOT the gate — three
        of four degenerate checkpoints sit at the opposite extreme, and a margin to this number passes
        them. The keys include the prefix, because the statistic does.
        """
        return math.log(max(1, self.reference.keys))

    def readable(self, model: nn.Module) -> AbstractContextManager[None]:
        """Read the encoder through the UNFUSED kernel, for the length of one forward pass and no longer.

        This is not a nicety. Transformers 5 serves `sdpa` by default, and SDPA's fused kernel never
        materialises the attention matrix — so `output_attentions=True` returns a tuple of `None`
        silently, and the very first real run of this gate died on its own refusal rather than reading
        stock. Training keeps the fused kernel; only the reading is eager, which is why it is a context
        manager and not a constructor flag.
        """
        if not hasattr(model, "set_attn_implementation"):
            return nullcontext()
        return _Readable(model)

    def maps(self, model: nn.Module) -> tuple[Float[Tensor, "b heads t t"], ...]:
        """The per-block attention maps, or a refusal if the backend served none.

        A fused-attention backend returns no maps rather than failing, which would turn this gate into a
        silent pass — the exact shape of failure it was built to end. `eager` is what makes the read
        possible; the refusal below is what happens when even that does not produce a matrix.
        """
        if self.probe is None:
            message = "the gate has no probe batch; a reference and a reading on different slices are not comparable"
            raise RuntimeError(message)
        with torch.no_grad(), self.readable(model):
            attentions = model(pixel_values=self.probe, output_attentions=True).attentions
        if not attentions or any(attention is None for attention in attentions):
            message = "the backbone returned no attention maps; the gate cannot pass on an unread encoder"
            raise RuntimeError(message)
        return tuple(attentions)

    def profile(self, model: nn.Module) -> AttentionProfile:
        """Both statistics for every block, from PATCH queries over ALL keys, on the HELD probe batch."""
        blocks = self.maps(model)
        return AttentionProfile(
            entropy=tuple(self.entropy(block) for block in blocks),
            prefix_mass=tuple(self.prefix_mass(block) for block in blocks),
            class_mass=tuple(self.class_mass(block) for block in blocks),
            keys=int(blocks[0].shape[-1]),
        )

    def entropy(self, attention: Float[Tensor, "b heads t t"]) -> float:
        """One block's mean entropy in nats over the raw softmax rows of the patch queries.

        Nothing is dropped and nothing renormalised: a patch-only view divides ~0 by ~0 exactly on the
        blocks where the class token has taken the mass, which is the failure worth seeing.
        """
        rows = attention[:, :, -self.tokens :, :].float()
        return float(-(rows * rows.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(dim=-1).mean())

    def prefix_mass(self, attention: Float[Tensor, "b heads t t"]) -> float:
        """How much of a patch query's mass lands on the prefix keys — the attention-sink statistic.

        The whole prefix rather than the class token alone, because DINOv3 prepends four registers and a
        sink can form on any of them; at DINOv2's prefix of one the two are the same number.

        That makes it the DETECTION statistic and not the diagnosis: it says a sink arrived, never which
        token took it, and on a register trunk those are opposite verdicts. `class_mass` is the other
        half, and neither is read alone.
        """
        prefix = attention.shape[-1] - self.tokens
        return float(attention[:, :, -self.tokens :, :prefix].float().sum(dim=-1).mean())

    def class_mass(self, attention: Float[Tensor, "b heads t t"]) -> float:
        """The same statistic over the CLASS TOKEN alone — key 0, never the registers behind it.

        Reported beside `prefix_mass` rather than instead of it, because the pair is the reading: their
        DIFFERENCE is the register mass, and a sink is benign exactly insofar as that difference carries
        it. Key 0 is the class token in both families — DINOv2 has nothing else in the prefix, and DINOv3
        prepends its four registers after it — so this indexes the same object on either trunk and needs
        no per-family branch.
        """
        return float(attention[:, :, -self.tokens :, :1].float().sum(dim=-1).mean())

    def degenerate(self, profile: AttentionProfile) -> bool:
        """Has ANY block dropped below the floor on either side of stock?

        Any block, not the top four together: the first version watched the top blocks because the only
        failure then known was a uniform run at the top, and D252's fronts start as low as block 5 —
        inside the half `unfreeze_last: 6` freezes, where the fine-tune has no gradient to repair it.
        """
        if len(profile.entropy) != len(self.reference.entropy):
            message = "the profile and its reference disagree on block count; the gate cannot compare them"
            raise RuntimeError(message)
        return any(share < self.floor for share in self.retention(profile))

    def report(self, profile: AttentionProfile) -> str:
        """The profile beside its own reference, so a log line carries the comparison it is judged on."""
        deviations = "  ".join(
            f"{value - stock:+.2f}" for value, stock in zip(profile.entropy, self.reference.entropy, strict=False)
        )
        kept = "  ".join(f"{share:.2f}" for share in self.retention(profile)) if profile.entropy else ""
        return f"{profile.report()} | vs stock {deviations} | keep {kept} (floor {self.floor:.2f})"
