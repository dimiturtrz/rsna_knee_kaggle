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
from collections.abc import Iterator
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
from kneemri.labels.targets import Targets
from kneemri.models.ema import DECAY, LiveWeights, WeightAverage
from kneemri.models.model import NO_LAYER_DECAY, StudyModel
from kneemri.training.accumulation import Accumulation
from kneemri.training.augmentation import INTENSITY_AUGMENT, Augmentation, AugmentConfig
from kneemri.training.heartbeat import Heartbeat
from kneemri.training.objective import Objective, Predicted
from kneemri.training.report_supervision import NoReportSupervision, ReportSupervision

logger = logging.getLogger(__name__)


WEIGHT_DECAY = 0.02
SEED = 2026


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

    layer_decay: float = NO_LAYER_DECAY
    """How much slower each block below the top one moves — the third member of the entangled triple.

    Defaults to no decay, which is what every ledger number was measured under. It belongs here rather
    than beside the schedule because it is the term that makes the OTHER two separable: at one flat rate
    a depth result and a rate result are the same measurement read twice.
    """


ONE_CYCLE_WARMUP = 0.15
"""Fraction of the run spent warming up before the decay — protects the pretrained encoder early."""


AUTOCAST = "autocast_dtype"
WINDOW = "slices_per_step"
READ_SIZE = "train_size"
AVERAGING = "weight_average"
SCHEDULED = "one_cycle"
REPORTS = "report_supervision"
ACCUMULATION = "micro_batches_per_step"
LAYER_DECAY = "encoder_rate_decay"
"""The keys `Trainer.policies` reports under — named here because the trainer is what they describe."""

NEEDED_FOR_A_RATIO = 2
"""Two backbone groups — below that there is no pair for a per-block ratio to be defined between."""


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
            self.adaptation.head_lr,
            self.adaptation.backbone_lr,
            self.adaptation.unfreeze_last,
            self.adaptation.layer_decay,
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
            LAYER_DECAY: self.encoder_rate_decay(),
        }

    def encoder_rate_decay(self) -> float:
        """The per-block rate ratio MEASURED off the optimiser, not read back off the flag that asked.

        This is the positive witness `--layer-decay` would otherwise lack. Everything else about the
        flag is silent: a decay that never reached `Adaptation` builds the flat two groups, trains
        perfectly well, and reports the flat recipe's number under the decayed arm's name — the exact
        failure shape `landed` exists for. The optimiser's own groups cannot lie about it.

        Read off the TOP TWO backbone groups because they are the pair the decay is defined between,
        and because `unfreeze_last` is clamped to the trunk's depth: counting groups instead would
        report a mismatch for `--unfreeze-last 24` that is only this check's own arithmetic.
        """
        backbone = self.optimiser.param_groups[1:]
        if len(backbone) < NEEDED_FOR_A_RATIO:
            return NO_LAYER_DECAY
        return float(backbone[-2]["lr"]) / float(backbone[-1]["lr"])

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

        `max_lr` is read back OFF THE OPTIMISER rather than rebuilt from the two `Adaptation` rates it
        used to name. Those two are the peak rates only while `parameter_groups` returns exactly two
        groups; under `--layer-decay` it returns one per opened block, and a two-element `max_lr` against
        seven groups is a ValueError at best and, had the count happened to match, a schedule that
        annealed the wrong block to the wrong rate. The peaks the optimiser was built with ARE the peaks.
        """
        self.schedule = torch.optim.lr_scheduler.OneCycleLR(
            self.optimiser,
            max_lr=[group["lr"] for group in self.optimiser.param_groups],
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
