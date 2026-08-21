"""Adapting the encoder and fitting the head, on report-derived targets weighted by confidence.

Two restraints on the encoder, both from the shape of the supervision rather than from taste:

**Only the last blocks move.** Early blocks of a vision transformer are generic edge and texture
filters; late blocks carry semantics. With 58 image-read labels and the rest derived from text, there
may not be enough supervision to improve the early ones and there is certainly enough to damage them.

**The encoder learns far more slowly than the head.** The head is random at initialisation and has
everything to learn; the encoder starts from a good solution and needs only to be moved off it. A single
learning rate either leaves the head untrained or destroys the encoder in the first few hundred steps.

Augmentation is rigid jitter plus intensity only — **no flips**. A horizontal flip would undo the
laterality normalisation that five of the twelve targets depend on, turning a medial meniscus into a
lateral one and teaching the model that the distinction is noise.
"""

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import UInt8
from numpy.typing import NDArray
from torch import Tensor

from kneemri.corpus.prefetch import Prefetch
from kneemri.corpus.slot_reader import GROUP
from kneemri.corpus.study_cache import CachedCorpus
from kneemri.device import VisibleCards
from kneemri.labels.targets import Targets
from kneemri.models.ema import DECAY, LiveWeights, WeightAverage
from kneemri.models.model import StudyModel
from kneemri.training.accumulation import Accumulation
from kneemri.training.augmentation import INTENSITY_AUGMENT, Augmentation, AugmentConfig
from kneemri.training.heartbeat import Heartbeat
from kneemri.training.objective import Objective, Predicted
from kneemri.training.report_supervision import NoReportSupervision, ReportSupervision

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

IT IS A 336 px CONSTANT, and `FITTED_SIZE` is now where that says so in code rather than only in
prose. No arm at 224, 392 or 448 recorded a peak (the log line postdates them), so the resolution axis
is DERIVATION: activations scale with the patch grid, so the slice cost goes as `size ** 2`.

The derivation is CONSUMED now, which it was not, and the reason is that leaving it in prose made the
cap silently wrong off 336 px rather than merely unproven. `needed()` took slices alone while
`--vram-gb` was being stripped from the arms, so `resolution.sh 420` would have asked for 16.28 GB
against a rung MEASURED at 23.42 — a cap under a demand, which this class's own docstring calls the
one thing that prevents nothing and guarantees the failure. The 420 arm could not have logged the peak
that was supposed to license the form, because it would have OOM'd before writing one."""

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

RESIDENT_GB = 0.22
"""The intercept of that line: what a step holds regardless of slice count — weights, optimiser state
and the allocator's own blocks, none of which the slice loop multiplies."""

WEIGHT_DECAY = 0.02
SEED = 2026


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
        step = (self.units / FITTED_UNITS) * (self.batch / FITTED_BATCH)
        return GB_PER_SLICE * (self.slices + 1) * step + RESIDENT_GB

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


@dataclass(frozen=True)
class Adaptation:
    """How much of the encoder moves, and how fast — one object because they cannot be read apart.

    The head is random at initialisation and has everything to learn; the encoder starts from a good
    solution and needs only to be moved off it. That argument sets the 125:1 default ratio, and it was
    inherited from the reference rather than measured.

    Depth and rate are ENTANGLED, which is why they stopped being three constants. Our bracket found
    unfreezing every block worth +0.0008, and a null result at a backbone rate of 8e-6 cannot
    distinguish "more trainable blocks do not help" from "no block was going to move anyway". Asking
    how many blocks should train is only a real question at a rate where a block moves.
    """

    head_lr: float = 1e-3
    backbone_lr: float = 8e-6
    unfreeze_last: int = 6
    """Six of twelve blocks — inherited from the reference, and only meaningful beside the rate above."""


ONE_CYCLE_WARMUP = 0.15
"""Fraction of the run spent warming up before the decay — protects the pretrained encoder early."""


AUTOCAST = "autocast_dtype"
WINDOW = "slices_per_step"
READ_SIZE = "train_size"
AVERAGING = "weight_average"
SCHEDULED = "one_cycle"
REPORTS = "report_supervision"
ACCUMULATION = "micro_batches_per_step"
"""The keys `Trainer.policies` reports under — named here because the trainer is what they describe."""


class Trainer:
    """Fits a `StudyModel` on a cached corpus and scores it on a held-out fold."""

    def __init__(
        self,
        model: StudyModel,
        device: torch.device,
        adaptation: Adaptation | None = None,
        augment_config: AugmentConfig | None = None,
        objective: Objective | None = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.adaptation = adaptation or Adaptation()
        self.augmentation = Augmentation(augment_config)
        self.objective = objective or Objective()
        groups = self.model.parameter_groups(
            self.adaptation.head_lr, self.adaptation.backbone_lr, self.adaptation.unfreeze_last
        )
        self.optimiser = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
        # fp16 until a run says otherwise, because that is what every number in the ledger was measured
        # under and a precision change is a numerics change.
        self.autocast_dtype = torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        self.schedule: torch.optim.lr_scheduler.OneCycleLR | None = None
        self.torch_generator = torch.Generator(device=device).manual_seed(SEED)
        # How many consecutive slices one read covers. Three is the RGB trick; anything else needs
        # per-slice pooling, because channel stacking has exactly three channels to stack into.
        self.window = GROUP
        # Attached rather than passed, for the same reason the schedule is: both are policies a run may
        # or may not want, and a sixth constructor argument trips the argument-count gate. The default
        # is a no-op object rather than None, so no call site has to test for one.
        self.average: LiveWeights = LiveWeights()
        # Attached for the same reason, and a no-op by default: `every=1` steps on every micro-batch,
        # which is the loop every number in the ledger was measured under.
        self.accumulation = Accumulation()
        self.heartbeat = Heartbeat()
        self.reports: NoReportSupervision = NoReportSupervision()
        # None means "read the cache at its own resolution", which is what every run did before
        # resampling existed. See `resample_to`.
        self.read_size: int | None = None

    def policies(self) -> dict[str, object]:
        """WHICH OPTIONAL POLICIES ARE ATTACHED, read off this trainer rather than off the run's flags.

        Each is applied by a method that a caller may simply never call, and the absence of every one of
        them is a working no-op rather than a crash — so a run missing one trains perfectly well and
        reports a number belonging to a recipe nobody asked for. `kneemri.training.landed` compares this
        against the request before the fitting starts.

        The LOSS switches come from the objective's own `settings()` rather than being re-derived here:
        which objective a run got is the objective's fact, and a trainer that inspected its type would
        be a second copy of the rule that chose it, free to disagree with `TrainingOptions.objective`.
        """
        return {
            **self.objective.settings(),
            INTENSITY_AUGMENT: self.augmentation.intensity(),
            AUTOCAST: self.autocast_dtype,
            WINDOW: self.window,
            READ_SIZE: self.read_size,
            AVERAGING: isinstance(self.average, WeightAverage),
            SCHEDULED: self.schedule is not None,
            REPORTS: isinstance(self.reports, ReportSupervision),
            ACCUMULATION: self.accumulation.every,
        }

    def report_supervision(self, reports: ReportSupervision) -> None:
        """Also supervise the study representation by its report embedding."""
        self.reports = reports

    def bfloat16(self) -> None:
        """Autocast in bf16 and drop the loss scaler with it.

        ATTACHED rather than passed, like every other thing a run may or may not want — the constructor
        is not where policies accumulate, and the argument gate said so at the sixth one.

        The scaler goes because bf16 carries fp32's exponent range, so there is no scale to collapse and
        nothing for the scaler to do: leaving it enabled would keep a second thing that can go wrong for
        no gain. Throughput is unchanged on this card — bf16 and fp16 share the same matmul path — so
        what this buys is robustness, not speed.
        """
        self.autocast_dtype = torch.bfloat16
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)

    def whole_stack(self, slices: int) -> None:
        """Read the ENTIRE cached stack in one step instead of one sampled window of it.

        Only meaningful with per-slice pooling, which encodes each slice alone and takes the
        per-dimension maximum — so a slot's embedding stops depending on which window was drawn, and a
        finding is no longer invisible unless it happens to fall inside the sampled three.

        The evidence for trying it is the strongest single number in this project: averaging ten
        overlapping windows at INFERENCE is worth +0.0156, far more than any method change measured
        here. Training already had that information available and was throwing it away each step.

        Costs one encoder pass per slice rather than per group — about 4x a step at twelve slices.
        """
        if not self.model.per_slice:
            message = f"whole-stack reading needs per-slice pooling, not {self.model.slice_pooling!r}"
            raise ValueError(message)
        self.window = slices

    def resample_to(self, size: int) -> None:
        """Train at `size` px from whatever resolution the cache holds, instead of the cache's own.

        Decoupling the two makes resolution a FREE variable: the encoder already interpolates in
        `slot_features`, so a 448 px cache can be read at 392 without a second 45-minute decode, and
        every size sees the same pixels through a different sampling grid rather than a different crop.

        It exists because resolution had a hard ceiling otherwise. Whole-stack at 448 px needs ~34 GB
        on a 32 GB card — tokens go as the square of the side — and every way of affording it changed
        a second variable: batch 8 halves the ranking term's within-batch pairs, fewer slices collides
        with the slice-count arm. Resampling moves the sampling grid and nothing else.

        The size is recorded in the member's config and fingerprint, because a member trained at 392
        from a 448 cache and rebuilt at 448 is a different function on different pixels — the same
        failure `slice_pooling` already caused once, caught then by the fingerprint gate.
        """
        self.read_size = size

    def exponential_average(self, decay: float = DECAY) -> None:
        """Keep an EMA of the weights, and score and save from it rather than from the live ones."""
        self.average = WeightAverage(self.model, decay)

    @contextmanager
    def evaluated(self) -> Iterator[None]:
        """Run a block under the weights this run would SAVE.

        Scoring the live weights while saving the averaged ones ranks epochs for a model that is never
        written to disk — the checkpoint's own number would then belong to a different function.
        """
        with self.average.applied(self.model):
            yield

    def accumulate(self, every: int) -> None:
        """Update on `every` micro-batches at once — see `Accumulation` for what that does and does not buy."""
        self.accumulation = Accumulation(every)

    def one_cycle(self, total_steps: int) -> None:
        """Attach a one-cycle schedule over the whole run, warming up then decaying to near zero.

        MEASURED REASON: at a constant learning rate our five folds peaked at epochs 13/13/10/12/11 and
        gained ~0 over their final five — saturated at ~12 epochs. A run that improves out to 60 epochs
        is not training longer, it is training on a decaying learning rate. `pct_start=0.15` spends the
        first sixth warming up, which also protects the pretrained encoder from the first few hundred
        steps at full rate.
        """
        self.schedule = torch.optim.lr_scheduler.OneCycleLR(
            self.optimiser,
            max_lr=[self.adaptation.head_lr, self.adaptation.backbone_lr],
            total_steps=max(total_steps, 1),
            pct_start=ONE_CYCLE_WARMUP,
        )

    def augment_bag(self, images: UInt8[Tensor, "b n g h w"]) -> UInt8[Tensor, "b n g h w"]:
        """Perturb a batch of slot bags with this run's own generator — see `Augmentation`."""
        return self.augmentation.bagged(images, self.torch_generator)

    def batches(
        self, positions: NDArray[np.int64], size: int, generator: np.random.Generator, *, shuffle: bool
    ) -> list[NDArray[np.int64]]:
        """Positions split into batches: shuffled for training, left in order for evaluation.

        TRAINING DROPS THE PARTIAL LAST BATCH. 3,498 studies over batches of 12 leaves six, and that
        six was the ONLY reason the encoder's graph was not static — a second shape means a second
        compilation, and it was why `compile_encoder` asked for dynamic shapes, which inductor cannot
        build here at all. One shape in, one graph out.

        The cost is bounded and known: the order is reshuffled every epoch, so the dropped six are a
        DIFFERENT six each time. Over twelve epochs a study is seen 11.98 times instead of 12. That is
        0.17% of the steps against a compiled encoder for all of them.

        Evaluation keeps every batch, because dropping studies there would not cost 0.17% of a
        gradient, it would silently score a different holdout than the one the ledger quotes.
        """
        order = generator.permutation(positions) if shuffle else positions
        batches = [order[start : start + size] for start in range(0, len(order), size)]
        if shuffle and len(batches) > 1 and len(batches[-1]) < size:
            batches.pop()
        return batches

    def plan(
        self,
        corpus: CachedCorpus,
        batches: list[NDArray[np.int64]],
        generator: np.random.Generator,
    ) -> list[tuple[NDArray[np.int64], int]]:
        """Every step of the epoch decided before the first one runs: its positions and its window start.

        The start is drawn per BATCH rather than per study, so a step is one window of the map rather than
        a gather across depths, and the randomness across steps still covers the stack.

        Drawing them all here rather than one per step is what lets the reader run ahead, and it draws the
        SAME numbers: these were the only draws left in the loop, one after another with nothing between,
        so pulling them forward reorders nothing.
        """
        return [(batch, int(generator.integers(0, corpus.slices - self.window + 1))) for batch in batches]

    def fit_epoch(
        self,
        corpus: CachedCorpus,
        targets: Targets,
        positions: NDArray[np.int64],
        size: int,
        generator: np.random.Generator,
    ) -> float:
        """One pass over the training positions, returning the mean weighted loss.

        The window start is drawn per BATCH rather than per study: every study in a step is read at the
        same depth, which keeps the batch a single contiguous slice of the memory map instead of a
        gather, and the randomness across steps still covers the stack.
        """
        # The fetch belongs to the corpus — `window()` reads only the slices asked for, where indexing
        # the batch and slicing afterwards materialises all twelve (97.5 MB per batch) to discard three
        # quarters of them: measured 35 ms against 3.3 ms, ten seconds of every forty-second epoch.
        self.model.train()
        # THE RUNNING TOTAL STAYS ON THE DEVICE. Reading a loss back with float() synchronises: the host
        # blocks until the backward that produced it has finished, so it cannot queue the next step while
        # the card works on this one — which is most of what the prefetch in front of the loop just bought.
        # The value is consumed twice, by the epoch mean and by a heartbeat line every thirty seconds, so
        # neither consumer wanted a per-step read. float64 makes the sum the one the host loop computed:
        # each float32 loss promoted and added, in step order.
        total = torch.zeros((), dtype=torch.float64, device=self.device)
        steps = 0
        plan = self.plan(corpus, self.batches(positions, size, generator, shuffle=True), generator)
        # TRUNCATED TO COMPLETE GROUPS before the reader is started, so the prefetch never reads a
        # micro-batch whose gradient the epoch will throw away. Without accumulation this is the
        # identity and the loop is the one every arm in the ledger ran.
        plan = plan[: self.accumulation.whole(len(plan))]
        planned = len(plan)
        # The gradient is zeroed ONCE per GROUP rather than per micro-batch, and that placement IS the
        # feature: gradients accumulate in the leaves between steps, so zeroing here and again after
        # each optimiser step is what makes a group sum instead of overwrite.
        self.optimiser.zero_grad(set_to_none=True)
        for batch, fetched in Prefetch(corpus, plan, self.window, pin=self.device.type == "cuda"):
            pixels = fetched.to(self.device, non_blocking=True)
            pixels = self.augment_bag(pixels)
            flags = torch.from_numpy(corpus.mask[batch]).to(self.device)
            score, weight = targets.rows(batch)
            with torch.autocast("cuda", dtype=self.autocast_dtype, enabled=self.device.type == "cuda"):
                # Both readouts from ONE encoder pass. The report head costs a small Linear and gets no
                # gradient when report supervision is off, so the loop needs no branch for it.
                logits, report = self.model.supervised(pixels, flags, self.read_size or corpus.image_size)
                judged = Predicted(
                    logits.float(),
                    torch.from_numpy(score).to(self.device),
                    torch.from_numpy(weight).to(self.device),
                )
                loss = self.objective(judged) + self.reports(report.float(), batch)
            self.scaler.scale(self.accumulation.share(loss)).backward()
            steps += 1
            # THE SCHEDULE AND THE WEIGHT AVERAGE MOVE WITH THE OPTIMISER, not with the reader. Stepping
            # a one-cycle schedule per micro-batch would run it `every` times too fast and raise past its
            # total mid-run, and an EMA updated between accumulations would average weights that never
            # took a step.
            if self.accumulation.closes(steps):
                self.scaler.step(self.optimiser)
                self.scaler.update()
                self.optimiser.zero_grad(set_to_none=True)
                if self.schedule is not None:
                    self.schedule.step()
                self.average.update(self.model)
            # The UNSHARED loss, so the epoch mean is the same number at any accumulation and the ledger
            # stays comparable across arms that accumulate and arms that do not.
            total += loss.detach().double()
            self.heartbeat.tick(
                steps,
                planned,
                # Bound as defaults rather than captured: the closure outlives the iteration only in
                # principle, but a loop variable read later is the classic way a log line reports the
                # wrong step, and the linter is right to refuse it.
                lambda accumulated=total, count=steps: float(accumulated) / count,
                int(pixels.shape[0] * pixels.shape[1] * pixels.shape[2]),
            )
        return float(total) / max(steps, 1)

    def predict(
        self,
        corpus: CachedCorpus,
        positions: NDArray[np.int64],
        size: int,
        generator: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Probabilities for a set of positions, read from the middle window of the stack.

        WHOLE-STACK RUNS HAVE NO MIDDLE: `window` is then every slice, `middle` is 0, and this reads
        exactly what the submission reads. That is what retired the epoch-selection mismatch — a windowed
        run picks its best epoch under a one-window policy while deployment averages ten, worth a uniform
        +0.0156 that no local number showed. For the members we ship the two policies are one policy.
        """
        self.model.eval()
        middle = (corpus.slices - self.window) // 2
        chunks = []
        # PADDED TO A FULL BATCH, then trimmed. The holdout cannot drop its remainder the way training
        # does — that would score a different holdout than the ledger quotes — so the last batch is
        # repeated up to size and the extra rows are discarded after the forward. The alternative is a
        # second graph shape for the sake of one batch per epoch.
        batches = self.batches(positions, size, generator, shuffle=False)
        kept = [len(batch) for batch in batches]
        plan = [
            (batch if len(batch) == size else np.concatenate([batch, np.repeat(batch[-1:], size - len(batch))]), middle)
            for batch in batches
        ]
        reader = Prefetch(corpus, plan, self.window, pin=self.device.type == "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=self.autocast_dtype, enabled=self.device.type == "cuda"):
            for (padded, fetched), rows in zip(reader, kept, strict=True):
                pixels = fetched.to(self.device, non_blocking=True)
                flags = torch.from_numpy(corpus.mask[padded]).to(self.device)
                scored = self.model(pixels, flags, self.read_size or corpus.image_size).float()
                chunks.append(torch.sigmoid(scored)[:rows])
        # ONE transfer after the loop rather than one per batch, for the reason the running total stays
        # on the device: `.cpu()` blocks the host until that batch's forward has finished, so the reader
        # thread and the card both wait on a copy of 12 floats per study. The whole holdout is 881x12.
        return torch.cat(chunks).cpu().numpy().astype(np.float64)

    def save_partial(self, destination: Path, payload: dict[str, object]) -> None:
        """Write the best weights so far, atomically, the moment they improve.

        Holding the only copy in memory until the final epoch is how a run that has already produced
        its best epoch loses everything — which the OneCycle off-by-one demonstrated by raising on the
        very last step of a thirty-epoch run. Temp-then-rename means a crash mid-write cannot leave a
        truncated file behind either.

        The payload is assembled by the CALLER, and is the same one the final save writes. This used
        to store only weights, holdout and epoch, which preserved a run's work but not its identity:
        a member with no fingerprint and no recorded reading is refused by MemberPackage, so a killed
        run left weights that could be inspected and never submitted.
        """
        partial = destination.with_suffix(".partial")
        temporary = partial.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(partial)
