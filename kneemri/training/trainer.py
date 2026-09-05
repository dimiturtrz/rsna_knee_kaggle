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
from typing import cast

import numpy as np
import torch
from jaxtyping import UInt8
from numpy.typing import NDArray
from torch import Tensor

from core.data.corpus.prefetch import Prefetch
from core.data.corpus.slot_reader import GROUP
from core.data.corpus.study_cache import CachedCorpus
from core.data.labels.targets import CONFIDENCE_FLOOR, Targets
from kneemri.models.ema import DECAY, LiveWeights, WeightAverage
from kneemri.models.encoder_surgery import NO_OFFSET
from kneemri.models.model import NO_LAYER_DECAY, StudyModel
from kneemri.training.accumulation import Accumulation
from kneemri.training.augmentation import (
    INTENSITY_AUGMENT,
    SLOT_DROPOUT,
    Augmentation,
    AugmentConfig,
)
from kneemri.training.grades import Predicted
from kneemri.training.gradient_clip import GradientClip
from kneemri.training.heartbeat import Heartbeat
from kneemri.training.mixup import Mixup
from kneemri.training.objective import Objective
from kneemri.training.report_supervision import NoReportSupervision, ReportSupervision
from kneemri.training.sampling import Sampling, Uniform
from kneemri.training.teacher_features import FEATURE_TEACHER, TeacherFeatures
from kneemri.training.teacher_targets import TEACHER, TeacherTargets

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
    weight_decay: float = WEIGHT_DECAY
    """How hard AdamW pulls every trained weight back toward zero — a member of the same entangled set.

    It was a module constant until 2026-08-31 and no arm had ever varied it. 0.02 came in with the
    reference recipe at a backbone rate of 8e-6; the rate sweep then moved the trunk five-fold to 4e-5
    without re-deriving it, so the decay every current number was measured under is the one chosen for a
    trunk that barely moved. Defaults to that constant, which is what keeps every existing arm identical.
    """

    unfreeze_last: int = 6
    """Six of twelve blocks — inherited from the reference, and only meaningful beside the rate above."""

    unfreeze_offset: int = NO_OFFSET
    """How many blocks at the TOP of the trunk stay frozen, which turns the suffix into a WINDOW.

    Zero for every run before 2026-09-05, and that is the suffix every ledger number was measured
    under. It exists because the block a run should open is decided by the KIND of distribution shift
    (Lee et al., ICLR 2023) and the input-level case — which natural-image pretraining to greyscale MRI
    is — wants EARLY blocks, the one thing `unfreeze_last` alone cannot express. See `BlockWindow`.
    """

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
BACKBONE_RATE = "encoder_top_rate"
MIXUP = "mixup_alpha"
OPTIMISER_DECAY = "optimiser_weight_decay"
WARMUP = "one_cycle_warmup"
CLIPPING = "gradient_clip_norm"
SAMPLING = "sampling_tilt"
SAMPLING_RAMP = "sampling_ramp"
CERTAINTY_RAMP = "certainty_ramp"
CORPUS_OVERSAMPLE = "corpus_oversample"
TARGET_CORPUS = "target_corpus"
"""The keys `Trainer.policies` reports under — named here because the trainer is what they describe."""

NEEDED_FOR_A_RATIO = 2
"""Two backbone groups — below that there is no pair for a per-block ratio to be defined between."""

NO_OPEN_BACKBONE = 0.0
"""The anchor of an empty backbone: under LoRA no base block is open, so no rate was asked for one."""


LOSS_QUANTILES = ("q25", "q50", "q75", "q90", "q95")
"""Which per-study loss quantiles the epoch record carries."""


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
        # THE OBJECTIVE'S OWN PARAMETERS ARE HANDED TO THE HEAD GROUP, and every term but one owns none.
        # `parameter_groups` collects the head BY EXCLUSION from the backbone, so a tensor the LOSS holds
        # would sit in no group at all: never stepped, left at its initialisation, and the arm would
        # report a fact about a random map — knee-7gyw's defect exactly, and silent.
        groups = self.model.parameter_groups(
            self.adaptation.head_lr,
            self.adaptation.backbone_lr,
            self.model.window(self.adaptation.unfreeze_last, self.adaptation.unfreeze_offset),
            self.adaptation.layer_decay,
            self.objective.attached(device),
        )
        self.optimiser = torch.optim.AdamW(groups, weight_decay=self.adaptation.weight_decay)
        # CAPTURED HERE rather than read live by `backbone_rate`, because `OneCycleLR` rewrites every
        # group's rate to `max_lr / div_factor` the moment it is built: a live read taken after the
        # schedule reports the anchor twenty-five times too small and the witness accuses a run whose
        # rate landed. The groups have just been built from `Adaptation`, so this is still a measurement
        # off the optimiser and not a read-back of the flag.
        self.backbone_anchor = cast("float", groups[-1]["lr"]) if len(groups) > 1 else NO_OPEN_BACKBONE
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
        # Attached and null by default for the same reason: alpha zero draws no blend at all, so the
        # loop is byte-for-byte the one every number in the ledger was measured under.
        self.mixup = Mixup()
        # Attached and null by default for the same reason: an unbounded step is what every number in
        # the ledger was measured under, and `GradientClip()` leaves `fit_epoch` doing exactly nothing.
        self.clipping = GradientClip()
        # Attached and null by default for the same reason, and here the default matters more than
        # anywhere else on this list: `Uniform` returns the positions untouched WITHOUT drawing from the
        # generator, so a run that asks for no sampling policy descends on exactly the stream every
        # number in the ledger was measured under.
        self.sampling: Sampling = Uniform()
        # A VALUE rather than a policy object, because there is nothing to dispatch on: the fraction is
        # read once, by `one_cycle`, as OneCycleLR's `pct_start`. It is a field rather than an argument
        # to that method so the request can be transcribed the same way every other switch is.
        self.warmup = ONE_CYCLE_WARMUP
        # A curriculum length, not a policy object: 0 means the table is read unchanged, and a positive
        # value blends it from the confidence floor toward full trust over that many epochs.
        self.certainty_ramp: int = 0
        # HOW FAR THE RAMP HAS COME, counted here rather than passed in: the caller already runs one
        # `fit_epoch` per epoch in order, so an argument would be a second copy of a number this object
        # can keep itself — and the copy is what a test stub gets wrong.
        self._epochs_fitted: int = 0
        self.heartbeat = Heartbeat()
        self.reports: NoReportSupervision = NoReportSupervision()
        # None rather than a no-op object, unlike every other policy here, because there is no
        # harmless zero for a teacher: a distilling objective handed an absent teacher must raise
        # rather than descend the label term alone under a distilling arm's name.
        self.teacher: TeacherTargets | None = None
        # None for the same reason, and a SECOND field rather than a widening of the one above: a run may
        # distil logits, features, both or neither, and folding them together would make "which of the
        # two arrived" unanswerable at exactly the moment `landed` asks it.
        self.feature_teacher: TeacherFeatures | None = None
        # None means "read the cache at its own resolution", which is what every run did before
        # resampling existed. See `resample_to`.
        self.read_size: int | None = None
        # Per-study losses for the CURRENT epoch; reset at the start of `fit_epoch` and turned into
        # quantiles at the end. Kept here rather than returned, so `fit_epoch`'s contract stays a scalar
        # and every existing caller is untouched.
        self._epoch_losses: list[NDArray[np.float64]] = []
        self._loss_quantiles: dict[str, float] = {}

    @property
    def loss_quantiles(self) -> dict[str, float]:
        """The last epoch's per-study loss quantiles, keyed q25/q50/q75/q90/q95.

        Empty before the first `fit_epoch`. NaN when no study contributed a loss, so a final fit that
        never calls `fit_epoch` still records a number of the right shape.
        """
        return self._loss_quantiles

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
            SLOT_DROPOUT: self.augmentation.config.slot_dropout,
            AUTOCAST: self.autocast_dtype,
            WINDOW: self.window,
            READ_SIZE: self.read_size,
            AVERAGING: isinstance(self.average, WeightAverage),
            SCHEDULED: self.schedule is not None,
            REPORTS: isinstance(self.reports, ReportSupervision),
            TEACHER: self.teacher is not None,
            FEATURE_TEACHER: self.feature_teacher is not None,
            ACCUMULATION: self.accumulation.every,
            LAYER_DECAY: self.encoder_rate_decay(),
            BACKBONE_RATE: self.backbone_rate(),
            MIXUP: self.mixup.alpha,
            OPTIMISER_DECAY: self.weight_decay(),
            WARMUP: self.warmup,
            CLIPPING: self.clipping.max_norm,
            SAMPLING: self.sampling.tilt(),
            SAMPLING_RAMP: self.sampling.ramp(),
            CERTAINTY_RAMP: self.certainty_ramp,
            CORPUS_OVERSAMPLE: getattr(self.sampling, "factor", 0.0),
            TARGET_CORPUS: (
                getattr(self.sampling, "target", "") if getattr(self.sampling, "factor", 0.0) > 0.0 else ""
            ),
        }

    def weight_decay(self) -> float:
        """The decay MEASURED off the optimiser, not read back off the flag that asked for it.

        The positive witness `--weight-decay` would otherwise lack, and the same argument
        `encoder_rate_decay` makes one line below: a decay that never reached `Adaptation` builds an
        optimiser at the inherited 0.02, trains perfectly well, and reports the old recipe's number under
        the new arm's name. Read off the HEAD group, which every configuration has — under LoRA the
        backbone groups are empty of base blocks, and the head is the one group `parameter_groups` always
        returns.
        """
        return float(self.optimiser.param_groups[0]["weight_decay"])

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

    def backbone_rate(self) -> float:
        """The ANCHOR the per-block decay is measured FROM, read off the optimiser rather than the flag.

        `encoder_rate_decay` above witnesses the RATIO between two backbone groups, and a ratio is blind
        to the rate it is a ratio of. A `--backbone-lr` that never reached `Adaptation` builds the groups
        at the inherited rate, decays them by exactly the 0.7 that was asked for, and reports every
        switch landed — so the flag with the largest effect on any foreign-trunk arm was the one flag
        with no positive witness at all.

        That is how `band_deep_matched` floored its training loss 0.189 above its own control with all
        four recipe flags recorded correct, and why its 0.8487 could not be read as a fact about depth:
        no number anywhere could say whether the rate compensation arrived (knee-62sol).

        Taken off the LAST backbone group, which `backbone_groups` builds at `layer_decay ** 0` — the
        anchor itself, and the only group whose rate is the flag rather than a power of it. Captured at
        construction for the reason spelled where it is captured: a schedule rewrites the live rate.
        """
        return self.backbone_anchor

    def report_supervision(self, reports: ReportSupervision) -> None:
        """Also supervise the study representation by its report embedding."""
        self.reports = reports

    def teach_from(self, teacher: TeacherTargets) -> None:
        """Carry an ensemble's readings of these same studies into every batch, for a distilling term."""
        self.teacher = teacher

    def teach_features_from(self, teacher: TeacherFeatures) -> None:
        """Carry a teacher's REPRESENTATIONS of these same studies into every batch, for a cosine term."""
        self.feature_teacher = teacher

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

        Meaningful for per-slice pooling, which encodes each slice alone and takes the per-dimension
        maximum — so a slot's embedding stops depending on which window was drawn, and a finding is no
        longer invisible unless it happens to fall inside the sampled three. And for a tubelet trunk,
        where it is the ONLY path that reaches one at all: the window is otherwise `GROUP` = 3, which
        no tubelet divides. `Model.check_whole_stack` owns which of the two applies.

        The evidence for trying it is the strongest single number in this project: averaging ten
        overlapping windows at INFERENCE is worth +0.0156, far more than any method change measured
        here. Training already had that information available and was throwing it away each step.

        Costs one encoder pass per slice rather than per group — about 4x a step at twelve slices.
        """
        self.model.check_whole_stack(slices)
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

    def blend(self, alpha: float) -> None:
        """Mix each batch with a permutation of itself — see `Mixup` for why this axis and not another epoch."""
        self.mixup = Mixup(alpha)

    def clip_gradients(self, max_norm: float) -> None:
        """Bound the norm of one optimiser step — see `GradientClip` for why this project never had one."""
        self.clipping = GradientClip(max_norm)

    def draw_with(self, sampling: Sampling) -> None:
        """Attach the policy that decides which studies each epoch descends on."""
        self.sampling = sampling

    def warm_up(self, fraction: float) -> None:
        """Spend this share of the one-cycle schedule warming up, instead of the inherited 0.15.

        BEFORE `one_cycle`, which is the one ordering constraint: the fraction is handed to OneCycleLR at
        construction and a scheduler already built cannot be told a different `pct_start`. `RunSetup`
        attaches every policy before it sizes the schedule, so the order holds there by construction.
        """
        self.warmup = fraction

    def ramp_certainty(self, epochs: int) -> None:
        """Spend this many epochs raising every cell's weight from the floor to its full trust."""
        self.certainty_ramp = epochs

    @staticmethod
    def _quantiles_of(batches: list[NDArray[np.float64]]) -> dict[str, float]:
        """Quantiles of the per-study losses collected across an epoch.

        NaN when no study contributed, so callers always get the same key shape. The quantiles are
        computed on CPU after the epoch, because the list is small and this keeps the training loop on
        the device.
        """
        if not batches:
            return {name: float("nan") for name in LOSS_QUANTILES}
        values = np.concatenate(batches)
        if len(values) == 0:
            return {name: float("nan") for name in LOSS_QUANTILES}
        levels = (0.25, 0.50, 0.75, 0.90, 0.95)
        return {name: float(level) for name, level in zip(LOSS_QUANTILES, np.quantile(values, levels), strict=True)}

    def _ramped_weights(self, weights: NDArray[np.float32], epoch: int) -> NDArray[np.float32]:
        """Blend cell weights from the confidence floor toward their full values.

        At epoch 0 every cell pulls with the floor; by epoch `certainty_ramp` the table is unchanged.
        The total mass is re-normalised to the full table's sum so the loss scale matches the un-ramped
        control — the objective's own divide-by-sum then sees the same denominator it would have seen.
        """
        if self.certainty_ramp <= 0:
            return weights
        alpha = min(epoch, self.certainty_ramp) / self.certainty_ramp
        full = weights.astype(np.float64)
        ramped = CONFIDENCE_FLOOR + (full - CONFIDENCE_FLOOR) * alpha
        # A batch can be all-floor at epoch 0, so the sum is never zero; clamp defensively.
        ramped = ramped * (full.sum() / max(ramped.sum(), 1e-12))
        return ramped.astype(np.float32)

    def one_cycle(self, total_steps: int) -> None:
        """Attach a one-cycle schedule over the whole run, warming up then decaying to near zero.

        MEASURED REASON: at a constant learning rate our five folds peaked at epochs 13/13/10/12/11 and
        gained ~0 over their final five — saturated at ~12 epochs. A run that improves out to 60 epochs
        is not training longer, it is training on a decaying learning rate. `pct_start` spends that share
        of the run warming up, which also protects the pretrained encoder from the first few hundred
        steps at full rate; it defaults to the inherited 0.15 and `warm_up` is what moves it.

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
            pct_start=self.warmup,
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
        # A fresh epoch's quantiles start empty; the list is populated per batch and reduced at the end.
        self._epoch_losses = []
        # THE RUNNING TOTAL STAYS ON THE DEVICE. Reading a loss back with float() synchronises: the host
        # blocks until the backward that produced it has finished, so it cannot queue the next step while
        # the card works on this one — which is most of what the prefetch in front of the loop just bought.
        # The value is consumed twice, by the epoch mean and by a heartbeat line every thirty seconds, so
        # neither consumer wanted a per-step read. float64 makes the sum the one the host loop computed:
        # each float32 loss promoted and added, in step order.
        total = torch.zeros((), dtype=torch.float64, device=self.device)
        steps = 0
        # THE DRAW COMES BEFORE THE SHUFFLE, and the two are different jobs: `sampling` decides WHICH
        # positions this epoch descends on, `batches` decides in what order. Under the default policy the
        # first is the identity and this line is the one every arm in the ledger ran.
        drawn = self.sampling.drawn(positions, generator)
        plan = self.plan(corpus, self.batches(drawn, size, generator, shuffle=True), generator)
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
            weight = self._ramped_weights(weight, self._epochs_fitted)
            # AFTER the augmenter and over all four tensors at once. Blending the raw read instead would
            # hand two studies the same warp and the same gain, which is a cheaper draw than it looks:
            # the pair would differ only in anatomy, not in appearance. Under the null policy every one
            # of these four calls is the identity — see `Blend`.
            blend = self.mixup.draw(len(batch), generator)
            pixels, flags = blend.images(pixels), blend.mask(flags)
            score, weight = blend.rows(score), blend.rows(weight)
            # AFTER the blend, and that order is forced: `Blend.mask` is an elementwise UNION, so a slot
            # withheld before it would be restored by any partner study that carries one. Dropout has to
            # be the last statement about what this step is allowed to see.
            pixels, flags = self.augmentation.dropped(pixels, flags, self.torch_generator)
            with torch.autocast("cuda", dtype=self.autocast_dtype, enabled=self.device.type == "cuda"):
                # Both readouts from ONE encoder pass. The report head costs a small Linear and gets no
                # gradient when report supervision is off, so the loop needs no branch for it.
                readout = self.model.supervised(pixels, flags, self.read_size or corpus.image_size)
                judged = Predicted(
                    readout.logits.float(),
                    torch.from_numpy(score).to(self.device),
                    torch.from_numpy(weight).to(self.device),
                    # INDEXED BY THE SAME `batch` the labels are, which is what makes the alignment a
                    # fact rather than a hope: both tables were reordered into the corpus's study order
                    # at load, so one index array addresses the same study in each.
                    # BLENDED TOO, by the same draw. A teacher table left un-mixed would score the
                    # blended pixels against the first study's soft labels alone, which is the distilling
                    # arm silently training on a different target than the label term does.
                    teacher=(
                        None
                        if self.teacher is None
                        else torch.from_numpy(blend.rows(self.teacher.rows(batch))).to(self.device)
                    ),
                    embedding=readout.context.float(),
                    # BLENDED BY THE SAME DRAW, for the reason the teacher table above is: an un-mixed
                    # target would pull the blended reading onto the first study's features alone, which
                    # is the feature arm silently descending toward a study it is not looking at.
                    teacher_embedding=(
                        None
                        if self.feature_teacher is None
                        else torch.from_numpy(blend.rows(self.feature_teacher.rows(batch))).to(self.device)
                    ),
                )
                loss = self.objective(judged) + self.reports(readout.report.float(), batch)
            # PER-STUDY LOSS for the epoch's quantiles: the same weighted BCE the scalar is built from,
            # averaged over labels within each study. Done under no_grad so it cannot feed back into
            # training, and on the UNSHARED loss so the distribution matches the number the epoch reports.
            with torch.no_grad():
                elementwise = torch.nn.functional.binary_cross_entropy_with_logits(
                    judged.logits, judged.scores, reduction="none"
                )
                weighted = elementwise * judged.weights
                per_study = weighted.sum(dim=1) / judged.weights.sum(dim=1).clamp(min=1e-6)
                self._epoch_losses.append(per_study.cpu().numpy().astype(np.float64))
            self.scaler.scale(self.accumulation.share(loss)).backward()
            steps += 1
            # THE SCHEDULE AND THE WEIGHT AVERAGE MOVE WITH THE OPTIMISER, not with the reader. Stepping
            # a one-cycle schedule per micro-batch would run it `every` times too fast and raise past its
            # total mid-run, and an EMA updated between accumulations would average weights that never
            # took a step.
            if self.accumulation.closes(steps):
                # AFTER the last micro-batch's backward and BEFORE the step: the norm has to be the one
                # the optimiser will actually apply, so clipping a partially accumulated gradient would
                # bound a quantity that is never used.
                self.clipping.applied(self.model, self.optimiser, self.scaler)
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
        self._loss_quantiles = self._quantiles_of(self._epoch_losses)
        self._epochs_fitted += 1
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
