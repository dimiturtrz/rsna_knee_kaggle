"""How much of the card one run may hold, and what a recipe will actually reserve.

A cap looks like a restriction and is a throughput fix: Windows pages GPU memory over PCIe rather than
raising OOM, so a run that does not fit does not fail — it crawls at full reported utilisation. The
whole cost model lives here rather than beside the training loop because it is a model of the CARD, and
because five separate arms have now died to a term of it being fitted on one configuration and handed
to another (knee-ylp0, knee-kwdh, knee-bj90, knee-yt35). It is the piece of this package that most wants reading
in one sitting.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from kneemri.device import VisibleCards

logger = logging.getLogger(__name__)

BYTES_PER_GB = 1024**3
"""So a VRAM budget can be stated in the unit the card is sold in."""

CONTENTION_POLL_SECONDS = 30.0
"""How often to re-ask the driver. Coarse on purpose: the thing being waited on is another training
run, which frees its card in one step at the end, not gradually."""

CONTENTION_WAIT_SECONDS = 1800.0
"""And how long that wait may last before the run starts anyway. Half an hour is longer than any epoch
here and shorter than a night, so a queue can absorb a co-tenant's arm without a person, and cannot be
held all night by one that hung."""

DEFAULT_SHARE = 0.8
"""Fraction of the card a run holds when it names no budget — read off the arms, not chosen: the
accepted ones cap at 20, 26 and 28 GB of 31.8, and 26 is 0.82."""

SMALLEST_RUN_GB = 5.5
"""Free VRAM below which no run this repo has ever recorded could have fitted, so starting is futile.

READ OFF OUR OWN RUNS rather than chosen: 224 epoch rows carry a `vram_peak_gb`, and they span 5.47 GB
(`dinov3_v3small_224_s2026` — the shallowest encoder at the smallest resolution we have fitted) to
26.24. A card with less free than the cheapest thing we know how to train cannot hold the dearest
either, and on Windows that does not raise — it pages over PCIe.

Those 224 rows recorded ALLOCATED under that name (see `EpochRow`), so 5.47 is the cheapest run's
demand and not what its allocator reserved — the true floor is that plus its fragmentation, i.e. this
number is low rather than high. Left at 5.5 until rows written under the corrected reading can say by
how much: a floor that is slightly too low lets a doomed run start, which the cap then answers with an
immediate OutOfMemoryError, where one guessed too high refuses runs that would have fitted."""

GB_PER_SLICE = 1.335
"""Reserved VRAM this recipe adds per slice — FITTED to our own epoch rows, not typed into an arm.

Every arm logs its own peak, and read across the slice axis at 336 px the peaks are a line: 12 slices
reserve 16.24 GB, 20 reserve 26.92. Those two give 1.335 a slice plus `RESIDENT_GB`, and the 16-slice
arm — held out of the fit — measured 21.57 against the line's 21.58. A slice is a whole forward and
backward through the encoder, so linearity in slice count is what the architecture predicts.

IT IS A 336 px CONSTANT, and `FITTED_SIZE` is where that says so in code rather than only in prose.
Activations scale with the patch grid, so the slice cost goes as the TOKEN count.

THE RESOLUTION AXIS IS NOW MEASURED, NOT DERIVED, AND IT HOLDS TO 2%. It was derivation when this form
landed, and `budget_residuals` grades it against every recorded peak — at fixed slices, batch, width
and trainable depth, the three points on the `train_size` axis are 336 px / 576 tokens (the fit),
392 px / 784 tokens (measured 22.43, derived 22.02, +1.8%) and 420 px / 900 tokens (measured 24.75,
derived 25.25, -2.0%). Two held-out points BRACKETING ZERO WITH OPPOSITE SIGNS over a 1.56x span in
tokens is what rules out the quadratic attention term this was once suspected of missing (knee-bj90):
SDPA never materialises the attention matrix, so memory is linear in tokens, and there is no t**2 term
to fit. Read residuals with `python -m kneemri.experiments.budget_residuals` before touching any
constant here — the ledger grades this file, and it does not agree with prose."""

FITTED_SIZE = 336
"""The pixel size `GB_PER_SLICE` was fitted at, so the scaling is exact there and only moves off it."""

FITTED_PATCH = 14
"""The patch the fit was taken at, because `size` is a proxy for TOKEN COUNT and not for pixels.

Every arm behind the fit is DINOv2, which is patch 14, so `size ** 2` and `(size / patch) ** 2` were
the same number up to a constant and the distinction cost nothing. DINOv3 is patch 16 and breaks the
identity: 224 px is 256 tokens under 14 and 196 under 16, so a pixel-only scaling overstates a v3 read
by a third. Named here so the ratio below reduces to 1 at the point it was fitted."""

FITTED_BATCH = 12
"""The batch `GB_PER_SLICE` was fitted at, so every arm at the settled recipe derives what it derived
before and only a run that moves the batch moves its own cap.

The term is LINEAR because activations are: a step holds `batch` independent stacks and the encoder's
weights are held once, and the weights are already the intercept (`RESIDENT_GB`). Like the size and
width terms above it is a derivation no recorded peak has checked off the fitted point, and like them
it is strictly better than the constant it replaces — a batch the caller lowered used to change the ask
by nothing at all."""

FITTED_DIM = 384
"""The feature width of the encoder `GB_PER_SLICE` was fitted on — `small`, and every arm behind it.

THE CAP WAS ENCODER-BLIND UNTIL 2026-08-20, and it cost a queued arm outright. `v3base_band` derived
12.68 GB from 20 slices at 224 px and died in seconds with `Tried to allocate 848.00 MiB ... 12.68 GiB
allowed` while 17.51 GiB of the card sat free. Nothing about that run was wrong except the cap: it was
a `small`-fitted number handed to a 768-wide encoder. This is the same failure `train_size or
image_size` fixed on the resolution axis (knee-ylp0), one axis further along — the cap saw the size the
step encodes at and still could not see WHICH ENCODER was encoding it.

Activation bytes go LINEARLY with width, which is why the term is a plain ratio rather than a fitted
one: every per-token tensor in a transformer block is `dim` wide (the projections, the residual stream,
the MLP's `4 * dim` hidden), and the attention matrix is `heads x tokens ** 2` with heads scaling with
width too — small has 6 and base 12. Nothing in a block is width-independent, so 768 costs twice 384.

It is a DERIVATION, exactly like the size term above and with the same standing: unchecked by any
recorded peak, and strictly better than the encoder-blind number it replaces. It is also biased the
safe way. Under LoRA the base weights are frozen and `--recompute-activations` discards most of what it
counts, so this OVERSTATES a v3 arm's real demand — and an overstated ask costs `await_headroom`
polling, where the understated one it replaces is an immediate OOM on a recipe that fits."""

FITTED_DEPTH = 12
"""How many blocks the fitted encoder runs. Every ViT this project has fitted is 12 deep — small, base
and both DINOv3 siblings — which is exactly why depth could hide inside `GB_PER_SLICE` for four months
and only announce itself when a 36-block convolutional trunk arrived."""

FITTED_UNITS = FITTED_DEPTH * (FITTED_SIZE / FITTED_PATCH) ** 2 * FITTED_DIM
"""The activation units of the fitted point — 12 blocks x 576 tokens x 384 wide.

THE DENOMINATOR OF THE WHOLE MODEL, written as the product it is so that changing any one factor of
the fitted point cannot leave a stale number behind. `needed()` is now a single ratio against this,
and every DINOv2 and DINOv3 arm derives the identical cap it derived before, because a 12-deep trunk's
units ARE `depth x (size / patch) ** 2 x dim` — the old formula, with the constant factor that used to
be assumed now said out loud."""

FITTED_TRAINABLE = 6
"""How many blocks carried a gradient at the fitted point — `Adaptation.unfreeze_last`'s default, and
every arm behind `GB_PER_SLICE`.

THE CAP WAS TRAINABLE-DEPTH-BLIND UNTIL 2026-08-21, and it is the FIFTH time a number fitted on one
configuration was handed to another. `encoder_adapt_thawed` asks for `--unfreeze-last 12`, derived 9.56
GB at batch 4 and died with `Tried to allocate 204.00 MiB ... 9.56 GiB allowed` while 9.55 GiB of the
card sat free — a cap that had scaled itself down correctly for the batch and then refused the run at
its own number, which is the signature of a term that is MISSING rather than mis-fitted.

Activation retention goes with the TRAINABLE blocks, not with the depth. A frozen block's activations
are freed the moment the next block has consumed them, because nothing downstream will ask for them
again; a trainable block's must survive until its backward. Under `--recompute-activations` what
survives is the checkpoint boundary rather than the whole block, but the count of boundaries is still
the count of blocks in the graph. So six blocks open and twelve open differ by a factor of two in the
term the slice loop multiplies, and `depth` cannot say it — `cnx_dense` runs 36 blocks with 6 open.

Like the size, width and batch terms it is a DERIVATION rather than a fit, and it reproduces the fitted
point exactly: a run that names nothing gets `unfreeze_last` 6 and a ratio of 1.

AND THE DERIVATION IS WRONG, MEASURABLY (knee-pf8b). A PURE ratio prices a frozen block at ZERO, but a
frozen block still runs its forward and still holds what the blocks above it read; only its gradients
and Adam state go away. `budget_residuals` reads three points off the ledger — 1.2142 a unit at four
blocks open, 1.3350 at six, 1.7730 at twelve — which a line `a + b*u` fits with a=0.9726, b=0.0604 and
which predicts the held-out twelve within 4.5%, where this ratio predicts it 51% over. In the other
direction it CAPPED `unfreeze4_s2026` at 11.79 GB against its own measured 14.79. Do not read the
`unfreeze_last` axis of the ledger as if this term were right."""

RESIDENT_GB = 0.22
"""The intercept of that line: what a step holds regardless of slice count — weights, optimiser state
and the allocator's own blocks, none of which the slice loop multiplies.

IT IS THE INTERCEPT AT `FITTED_PARAMS_M`, and reading it as a universal constant refused an arm
outright. See `OPTIMISER_GB_PER_MPARAM` for the term that carries it off that point."""

FITTED_PARAMS_M = 22.0
"""The trunk size the intercept was fitted on — DINOv2-small, and every arm behind the fit."""

OPTIMISER_GB_PER_MPARAM = 0.016
"""What each million parameters ABOVE the fitted point costs, in gigabytes — physics, not a fit.

A parameter under AdamW costs 16 bytes that no slice loop multiplies: 4 for the fp32 master weight, 4
for its gradient, 8 for the two moments. Autocast changes the ACTIVATIONS to bf16 and leaves that
sixteen alone. So a million parameters is 0.016 GB, and the 64M by which a ViT-B exceeds `small` is a
whole gigabyte the cap could not see.

IT COST `v3base_dense` OUTRIGHT on 2026-08-20 — `9.17 GiB is free; 21.68 GiB allowed`, a run refused
while nine gigabytes sat unused, and then an idle hour behind a watcher that had already fired
(knee-bj90). This is the term that was missing, and it is the FOURTH time a cap fitted on one
configuration was handed to another.

ADDED ABOVE THE FITTED INTERCEPT rather than replacing it, because the two quantities are not the same
kind. `RESIDENT_GB` is MEASURED at 22M and includes the allocator's own blocks, which do not scale with
anything; the excess over it is derived. Written this way the fitted point reproduces exactly — every
recorded DINOv2-small arm derives the number it derived — and only a bigger trunk moves."""


@dataclass(frozen=True)
class DeviceBudget:
    """The card a run may use, and how much of it it is allowed to hold.

    A cap looks like a restriction and is actually a throughput fix, which is why it is a first-class
    object rather than three lines in `main`. MEASURED 2026-08-11 on a 32 GB 5090, whole-stack reading
    at batch 12:

        uncapped   32.0 GB held, 145 W of 575, 100% "utilisation", NO epoch in fifteen minutes
        18 GB cap  19.2 GB held, 470-514 W,     99% utilisation,    training normally

    Windows pages GPU memory over PCIe rather than raising OOM, so a run that does not fit does not
    fail - it crawls, while `nvidia-smi` reports full utilisation because a kernel is resident. Power
    draw is the honest signal: a card doing real work draws its budget. Holding the run to a share of
    the card keeps it inside physical VRAM, and turns the worst failure mode we have into an immediate
    OutOfMemoryError.

    The second reason is coexistence: another project trains on this same card, and an uncapped run
    takes all of it.
    """

    vram_gb: float = 0.0
    """What the run ASKED for. 0 means it named nothing, and then the recipe's own need applies."""

    slices: int = 0
    """How many slices a step holds — the field the reserved peak is most of a function of. 0 means the
    caller did not say, and then only the card-fraction default is available."""

    units: float = FITTED_UNITS
    """Summed tokens-times-width over every block of the trunk, at the size this run reads — what the
    caller gets from `Encoder.activation_units`.

    IT REPLACED THREE FIELDS (size, patch, dim) THAT COULD NOT SAY WHAT THEY MEANT. Each of them was a
    factor of the same quantity, and holding them separately forced this class to reassemble it with a
    formula — `(size / patch) ** 2 * dim` — that silently assumed every block of every trunk sees the
    final grid at the final width. That is true of a vision transformer and false of everything else,
    which is why the axis had to move out of here and into the encoder, where the block structure
    already lives. Defaults to the fitted encoder's own number, so a caller that says nothing derives
    exactly what it derived before."""

    batch: int = FITTED_BATCH
    """How many studies a step holds at once — the one factor the fit had baked in as a constant.

    MEASURED 2026-08-20: `base_dense` at the settled recipe derived 56.3 GB against 30.3 free and parked
    for the full contention bound without starting a step. Halving `--batch` to 6 derived 56.3 AGAIN,
    because batch was invisible here — so the only lever a caller has against an overstated ask moved
    nothing, and the larger-encoder axis was unreachable rather than expensive."""

    trainable_blocks: int = FITTED_TRAINABLE
    """How many encoder blocks carry a gradient — `unfreeze_last`, clamped to the trunk's own depth.

    It is the axis `units` cannot express. `units` is a property of the TRUNK at a size; this is a
    property of what the RECIPE opened, and the two move independently — a 36-block ConvNeXt with six
    blocks open holds less than a 12-block ViT with all twelve open. See `FITTED_TRAINABLE`."""

    params_m: float = FITTED_PARAMS_M
    """The trunk's parameter count in millions — what the OPTIMISER's share of the bill is a function of.

    The other three fields together describe ACTIVATIONS, which the slice loop multiplies. This one
    describes what it does not: weights, gradients and Adam's two moments, held once per step whatever
    the slices are. Defaults to the fitted trunk, so a caller that says nothing derives what it did."""

    def needed(self) -> float:
        """What this recipe will actually reserve, plus one slice of slack — 0 if the slices are unknown.

        THE SLACK IS ONE SLICE rather than a round number, because that is the granularity at which
        this quantity moves: nothing here can overshoot by less than a slice, so a margin below one is
        a margin against nothing, and a margin above one is paying for a slice that cannot arrive. The
        fit's own residual does not need covering — it is 0.01 GB on a held-out point. It is one slice
        AT THIS SIZE, which is what the square term gives for free.

        A SLICE COSTS ITS TOKEN COUNT TIMES ITS FEATURE WIDTH, and both ratios are exact at the fitted
        point by construction — 336 px, patch 14, 384 wide reduces to `GB_PER_SLICE`, so no arm behind
        the fit sees a different number than before, and neither does any other DINOv2 run at any size.
        `(size / patch) ** 2` is the same quantity the old `size ** 2` was measuring, written so that
        it stays true when the patch moves: it is the token count, and a pixel-only form silently
        overstates a patch-16 read by (16/14)**2. Off the fitted point the term is a derivation the
        arms have not checked yet, and it is
        still strictly better than the alternative it replaces: 420 px derives 27.34 GB against a rung
        measured at 23.42, i.e. it OVERSTATES, where ignoring the size understated by 7 GB. An
        overstated ask costs `await_headroom` polling; an understated one is an immediate OOM on a
        recipe that fits.
        """
        if not self.slices:
            return 0.0
        step = (self.units / FITTED_UNITS) * (self.batch / FITTED_BATCH) * (self.trainable_blocks / FITTED_TRAINABLE)
        optimiser = RESIDENT_GB + OPTIMISER_GB_PER_MPARAM * max(0.0, self.params_m - FITTED_PARAMS_M)
        return GB_PER_SLICE * (self.slices + 1) * step + optimiser

    def capped(self, whole: float) -> float:
        """How much of a card this run may hold — what it asked for, or the default share of it.

        THE DEFAULT USED TO BE NO CAP, which made the class's own worst failure mode the behaviour a run
        got by saying nothing. Every accepted arm names `--vram-gb` (20, 26 or 28 of 31.8), so uncapped
        was never what a measured run did — it was only what a smoke, a bisect or a one-off got, and
        those are precisely the runs nobody watches the power draw of.

        `DEFAULT_SHARE` is not a chosen number: it is where those arms landed. 26 of 31.8 is 0.82 and 28
        is 0.88, and the cap bounds PYTORCH'S allocator rather than the device — driver context, the
        desktop and the co-tenant all live outside it, which is why `lora.sh` measured 30 as WORSE than
        26. So the default sits at the bottom of the band the arms converged on, where being wrong costs
        an immediate OutOfMemoryError instead of a silent crawl.

        THE RECIPE'S OWN NEED COMES FIRST, and it is why an arm should stop naming a number at all. A
        hand-typed cap is a THIRD statement of the same fact — the recipe says 12 slices, the run
        reserves 16.24 GB, and the arm says 28 — and the third one goes stale silently: thirteen arms
        overstate by 1 to 12 GB, `depth_regions_narrow` because it was `depth_regions` with the slices
        dropped and the 20-slice cap copied along. Slack is not free once a co-tenant holds the card:
        `await_headroom` blocks until `free >= vram_gb`, so an overstated ask makes a run poll for
        memory it will never allocate, burn the wait bound, and then start into the PCIe crawl.

        THE SHARE IS THE FALLBACK, NOT A CEILING OVER THE DERIVATION. Clamping the derived need to
        `DEFAULT_SHARE` was the first version of this and it was wrong: 20 slices reserve 26.92 GB and
        the share is 25.44, so every 20-slice arm — recipes this repo has RUN, at a named 28 — would
        have been capped below its own measured demand and OOM'd on the spot. A cap under a measured
        need prevents nothing and guarantees the failure. The share answers the different question of
        what a run holds when NOTHING is known about it, which is the only case it now applies to.

        A need above the card is left to raise. That is the documented preference of this whole class:
        an immediate OutOfMemoryError over a silent PCIe crawl.
        """
        return self.vram_gb or self.needed() or DEFAULT_SHARE * whole

    def floor(self) -> float:
        """Free VRAM below which this run refuses to start at all — not the same number as its CAP.

        THE CAP IS THE WRONG TEST for refusing. It defaults to 0.8 of the card, so refusing on `free <
        capped` would refuse every time the co-tenant holds a fifth of the card — which is most of the
        time, and mostly harmless: a 5.5 GB smoke does not care that 25 GB is unavailable. That test
        belongs to the WARNING, and stays there.

        What is fatal is free VRAM below what our cheapest run has ever needed, because then no recipe
        fits and Windows will not say so. A run that named `--vram-gb` has stated its own need, and it
        is taken as the floor when it is SMALLER than the corpus-derived one — a run claiming to fit in
        3 GB is refused at 3, not at 5.5.
        """
        return min(self.vram_gb or SMALLEST_RUN_GB, SMALLEST_RUN_GB)

    def headroom(self) -> tuple[float, float]:
        """Free and total VRAM in GB, as the driver sees it right now — (0, 0) with no visible card.

        What is FREE is not what the card HAS, and the difference is another process. On 2026-08-17 a
        60-study smoke that normally takes seconds crawled at 12 img/s and then died at step 4 with no
        traceback; the card was at 32052 MiB of 32607 because a second project was training on it, and
        three `nvlddmkm` engine resets in four minutes took both runs down. Every line in this class's
        docstring already described that failure and the run still had no way to SAY it, because a cap
        is arithmetic against the total and the total does not move when someone else takes the card.
        """
        if not torch.cuda.device_count():
            return 0.0, 0.0
        free, whole = torch.cuda.mem_get_info(torch.cuda.current_device())
        return free / BYTES_PER_GB, whole / BYTES_PER_GB

    def await_headroom(self, sleep: Callable[[float], None] = time.sleep) -> tuple[float, float]:
        """Wait, up to a bound, for the co-tenant to give the card back. Returns the headroom it starts on.

        WAITING RATHER THAN REFUSING, because the other project on this card is not a fault to report —
        it is a second real run belonging to the same person, and it finishes. A queue that refuses on
        contention loses the arm; a queue that waits loses minutes. The bound is what keeps the second
        case from becoming the first: after it, the run starts anyway and says what it is starting into,
        so an overnight queue cannot be stalled indefinitely by a job nobody is watching.

        WAITING RATHER THAN PROCEEDING, because proceeding is the failure this whole class documents.
        Windows pages over PCIe instead of raising, so a run that starts into a held card does not fail
        — it crawls at a tenth of the throughput, and the pair of us then draws enough to take the
        driver down: three `nvlddmkm` engine resets in four minutes on 2026-08-17, which killed both
        projects' processes, not just the one that was late.

        ONLY A RUN WITH A STATED NEED WAITS, and the recipe states one as well as the flag does. It
        used to be the flag alone, on the reasoning that an uncapped run's "need" is the whole card and
        the whole card is never free, so waiting would stall it for the full bound and buy nothing.
        That holds for a run that named nothing at all and no longer holds for one whose slice count
        says what it will reserve: 16.24 GB IS free most of the time, so waiting for it is a wait that
        ends. A run with neither still does not wait; `select` warns there, as it did before.
        """
        waited = 0.0
        free, whole = self.headroom()
        wanted = self.vram_gb or self.needed()
        while free < wanted and waited < CONTENTION_WAIT_SECONDS:
            logger.info(
                "waiting for VRAM: %.1f GB free, this run wants %.1f — %.0f s of %.0f",
                free,
                wanted,
                waited,
                CONTENTION_WAIT_SECONDS,
            )
            sleep(CONTENTION_POLL_SECONDS)
            waited += CONTENTION_POLL_SECONDS
            free, whole = self.headroom()
        return free, whole

    def select(self) -> torch.device:
        """The device to train on, already held to this budget.

        The choice itself is `VisibleCards`'s — see there for why availability is the wrong question.
        What stays here is the BUDGET, which is meaningless without a card.
        """
        device = VisibleCards().device()
        if device.type != "cuda":
            return device
        free, whole = self.await_headroom()
        # REFUSING, PAST THE WAIT. The card has been short for the full bound and is short of the
        # cheapest run on record, so there is no recipe left that fits — and the one thing this must not
        # do is start, because Windows answers an over-commit by paging over PCIe rather than raising.
        # That is what took three nvlddmkm engine resets and BOTH projects' processes down on
        # 2026-08-17: the run did not fail, it crawled, and the pair of us drew enough to reset the card.
        if free < self.floor():
            message = (
                f"{free:.1f} GB of {whole:.1f} GB VRAM free, below the {self.floor():.1f} GB this run "
                f"needs — another process holds the card. Refusing rather than paging over PCIe."
            )
            raise RuntimeError(message)
        budget = self.capped(whole)
        # ONLY REACHABLE PAST THE WAIT'S BOUND, so this is no longer "the card is busy" — it is "the
        # card stayed busy for half an hour and this run is starting into it anyway". Named before the
        # first batch, never inferred from the throughput afterwards, because the symptom is not a
        # failure: it pages over PCIe and reads as a slow recipe.
        if free < budget:
            logger.warning(
                "VRAM: %.1f GB free of %.1f GB — this run wants %.1f and another process holds the rest",
                free,
                whole,
                budget,
            )
        # An INDEX, not torch.device("cuda"): both torch calls reject an indexless device.
        card = torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(min(budget / whole, 1.0), card)
        logger.info("VRAM capped at %.1f GB of %.1f GB, %.1f free", budget, whole, free)
        return device
