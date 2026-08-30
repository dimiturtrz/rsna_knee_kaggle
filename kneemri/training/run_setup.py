"""What a run's flags BUILD, separated from what they say.

`TrainingOptions` answered two different questions and was measured doing it: LCOM4 = 5, the lowest
cohesion of any class here, and the one cluster that never touched the others was this — the seven
methods that turn flags into the objects a fit runs on. The other clusters read a flag pair as a value
(`crop`, `band`, `reading`) or read the corpus (`cache`, `sampled`); none of them constructs anything.

Splitting them also puts train.py back under the god-file ceiling it had crept to within 44 lines of,
which is the honest reason the cluster kept growing there: every new switch is one more `if self.x:`
in setup, and setup was living in the module whose job is the command line.

WHAT DOES NOT MOVE: `supervision` reads two flags into a value the LABEL table takes, which is the
`crop`/`band` shape, not this one.
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from kneemri.models.backbone import ENCODERS, Encoder
from kneemri.models.model import StudyDesign, StudyModel
from kneemri.training.augmentation import AugmentConfig
from kneemri.training.device_budget import DeviceBudget
from kneemri.training.grades import Grades, OrdinalGrades
from kneemri.training.label_focus import LabelFocus
from kneemri.training.landed import Request
from kneemri.training.objective import (
    ContrastiveObjective,
    DecorrelatedObjective,
    DistilledObjective,
    LabelStructure,
    MarginObjective,
    Objective,
    RankedObjective,
    Shaping,
)
from kneemri.training.options import TrainingOptions
from kneemri.training.pretrained_encoder import PretrainedEncoder
from kneemri.training.remembered_contrast import RememberedContrast
from kneemri.training.remembered_ranking import RememberedRanking
from kneemri.training.report_supervision import ReportSupervision
from kneemri.training.side_ranking import PairedSides, SideRanking
from kneemri.training.teacher_targets import TeacherTargets
from kneemri.training.trainer import Adaptation, Trainer

logger = logging.getLogger(__name__)

ENCODER_FULLY_OPEN = 1_000
"""More blocks than any trunk here runs, so `min` with the encoder's depth opens all of them.

Under LoRA `unfreeze_last` describes BASE weights, which are frozen, while the adapters it says nothing
about sit in every block and retain their activations. So the honest trainable count is the whole trunk,
and this is how the clamp says that without naming a depth that would go stale."""


@dataclass(frozen=True)
class RunSetup:
    """The model, the trainer and the policies one run's flags ask for.

    Everything here is checkable without a card, a corpus or a downloaded encoder, which is the point:
    a miswired switch that surfaces after forty minutes of queue costs the slot, and every one of these
    methods exists because a run once trained the plain recipe under an arm's name.
    """

    options: TrainingOptions

    def adaptation(self) -> Adaptation:
        """How the encoder adapts: which blocks move and how fast, which are one decision."""
        return Adaptation(
            head_lr=self.options.head_lr,
            backbone_lr=self.options.backbone_lr,
            unfreeze_last=self.options.unfreeze_last,
            layer_decay=self.options.layer_decay,
        )

    def budget(self) -> DeviceBudget:
        """What this run will hold, assembled from the flags and the encoder they name.

        IT LIVES HERE BECAUSE TWO CALLERS NEED THE SAME ANSWER and only one of them is the trainer. The
        other is the queue: `lib.sh` refuses to start an arm until the card has room for it, and it did
        that against a hard-coded 18432 MiB — a figure derived from a 12-slice arm and then handed to
        every arm, which is the same fitted-on-one-configuration failure `DeviceBudget` itself has now
        recorded five times. A batch-2 arm that reserves 9.6 GB waited for 18.

        `Encoder` answers the three questions the cap cannot ask a flag: how many activation units the
        trunk holds at this run's size, how many of them the OPEN blocks hold, and how many parameters
        the optimiser has to carry. `opened` is what supervision opened — `unfreeze_last`, except under
        LoRA, where every block carries an adapter and retains, so the base-weight count would
        understate it by the full depth. The CLAMP against the trunk's real block count is the
        encoder's, not this method's: `depth` is 12 on a trunk that runs 36.
        """
        encoder = ENCODERS[self.options.variant]
        size = self.options.train_size or self.options.image_size
        opened = ENCODER_FULLY_OPEN if self.options.lora_rank else self.options.unfreeze_last
        return DeviceBudget(
            self.options.vram_gb,
            self.options.slices,
            encoder.activation_units(size),
            self.options.batch,
            retained=encoder.retained_units(size, opened),
            params_m=encoder.params_m,
        )

    def requested(self, device: torch.device) -> Request:
        """These flags in the vocabulary the pre-flight check compares them in, transcribed ONCE.

        The transcription was seventeen `field=arguments.field` lines inside `main`, and that is one
        copy-paste away from the defect the check itself exists to catch: `weight_floor=
        arguments.calibrated_targets` parses, type-checks, runs, and leaves the guard agreeing with the
        wrong flag — a run would then be cleared for a switch nobody asked for and refused for one it
        did. Here it is a method a test can walk field by field without a card or a corpus.

        THE THREE NAMES THAT DO NOT MATCH are the three that are not a flag read straight off: two are
        an argument's PRESENCE, and `compiling` is resolved against the DEVICE because `model` compiles
        on CUDA only — on CPU the honest request is False, and a check reading the flag alone would
        fail every smoke run on this machine.
        """
        options = self.options
        return Request(
            lora_rank=options.lora_rank,
            slice_regions=options.slice_regions,
            recompute=options.recompute_activations,
            compiling=options.compile_encoder and device.type == "cuda",
            bf16=options.bf16,
            whole_stack=options.whole_stack,
            slices=options.slices,
            train_size=options.train_size,
            ema=options.ema,
            one_cycle=options.one_cycle,
            accumulate=options.accumulate,
            reports=options.report_vectors is not None,
            rank_weight=options.rank_weight,
            distill_weight=options.distill_weight,
            dissent_weight=options.dissent_weight,
            contrast_weight=options.contrast_weight,
            auc_margin=options.auc_margin,
            teaching=options.teacher_targets is not None,
            pair_memory=options.pair_memory,
            per_label_loss=options.per_label_loss,
            normalise_by_count=options.normalise_by_count,
            intensity_aug=options.intensity_aug,
            weight_floor=options.weight_floor,
            calibrated_targets=options.calibrated_targets,
            finding_calibration=options.finding_calibration,
            severity_weight=options.severity_weight,
            hold_annotations=options.hold_annotations,
            grade_ladder=options.grade_ladder,
            side_weight=options.side_weight,
            # The EMPTY STRING and not None, because that is what `LabelFocus.settings()` reports for
            # the null policy, and this comparison is a strict equality against exactly that report.
            focus_label=options.focus_label or "",
            unfreeze_last=options.unfreeze_last,
            layer_decay=options.layer_decay,
            mixup=options.mixup,
        )

    def objective(self) -> Objective:
        """The loss policy these flags ask for.

        Off the command line rather than inside `main` so the choice can be checked without a GPU, a
        corpus or a downloaded encoder. The alternative is finding out that a queued arm was miswired
        after it has already waited forty minutes for the card.
        """
        options = self.options
        self._check_composed()
        structure = LabelStructure(
            grades=OrdinalGrades() if options.grade_ladder else Grades(),
            sides=PairedSides(options.side_weight) if options.side_weight else SideRanking(),
            focus=LabelFocus(options.focus_label),
        )
        shaping = Shaping(
            normalise_by_count=options.normalise_by_count,
            per_label=options.per_label_loss,
            structure=structure,
        )
        if options.auc_margin:
            return MarginObjective(options.auc_margin, shaping)
        if options.dissent_weight:
            return DecorrelatedObjective(options.dissent_weight, shaping)
        if options.distill_weight:
            # BEFORE the ranking branch, and carrying `rank_weight` through rather than excluding it:
            # the owner's design is one training that runs both terms, so this is the composed objective
            # rather than a third alternative to the other two.
            return DistilledObjective(options.distill_weight, shaping, rank_weight=options.rank_weight)
        if options.contrast_weight:
            return self._contrastive(shaping)
        if options.pair_memory:
            return RememberedRanking(options.rank_weight, shaping, capacity=options.pair_memory)
        if options.rank_weight:
            return RankedObjective(options.rank_weight, shaping)
        return Objective(shaping)

    def _contrastive(self, shaping: Shaping) -> ContrastiveObjective:
        """The contrastive objective these flags ask for, widened by a remembered pool or not.

        A METHOD RATHER THAN TWO MORE BRANCHES IN `objective`. That chain sits at the complexity ceiling
        already, and the choice here is not a fifth alternative to the other terms — it is one term with
        one option, so it reads as one branch up there and its own question down here.

        `rank_weight` is carried through in both, for `DistilledObjective`'s reason: the house recipe
        passes --rank-weight 1.0, so excluding it would make every contrastive arm a two-variable change
        against the control it is measured on.
        """
        options = self.options
        if options.pair_memory:
            return RememberedContrast(
                options.contrast_weight,
                shaping,
                rank_weight=options.rank_weight,
                capacity=options.pair_memory,
            )
        return ContrastiveObjective(options.contrast_weight, shaping, rank_weight=options.rank_weight)

    def _check_composed(self) -> None:
        """Every way the loss flags fail to compose, raised before the card is touched.

        SEPARATE from `objective` because the two answer different questions — which objective, and
        whether the flags asking for it can be honoured together — and because holding both put the
        dispatch over the complexity ceiling, which is the gate saying the same thing.
        """
        options = self.options
        # FIRST of the guards, ahead of the dissent one for its own reason: this flag replaces the term
        # every other objective here is built on, so a composition it cannot honour has to be named
        # against THIS flag rather than against whichever term was stacked on top of it.
        if options.auc_margin:
            self._check_margin_alone()
        # A dissenting arm that also passed `--pair-memory` would otherwise be told about a missing
        # `--rank-weight` it never intended to give, which sends the reader to the wrong flag entirely.
        if options.dissent_weight:
            self._check_dissent_alone()
        if options.contrast_weight:
            self._check_contrast_alone()
        if options.pair_memory and not (options.rank_weight or options.contrast_weight):
            # LOUD, for the reason `reports` gives below: a memory with no term to feed is a silent
            # no-op that trains the plain recipe under the arm's name, and would be read as a
            # refutation of the mechanism rather than of the invocation.
            message = (
                f"--pair-memory {options.pair_memory} widens the pool of a ranking or a contrastive "
                "term and neither --rank-weight nor --contrast-weight was given, so there is nothing "
                "for it to widen"
            )
            raise ValueError(message)
        if options.distill_weight and options.pair_memory:
            # LOUD for the same reason as above. `RememberedRanking` widens the LABEL term's pair pool
            # and knows nothing about a teacher, so the two together would train a distilling arm whose
            # remembered pairs silently carry no teacher at all — half the mechanism, under its name.
            message = (
                f"--distill-weight {options.distill_weight} and --pair-memory {options.pair_memory} are "
                "not composed: the remembered pool widens the label term only, so the teacher would be "
                "dropped for every remembered step"
            )
            raise ValueError(message)

    def _check_margin_alone(self) -> None:
        """The four flags `--auc-margin` cannot be composed with, each LOUD for `objective`'s reason.

        Every other objective in this file's chain is cross-entropy PLUS a term, and this one removes
        cross-entropy. So each of those flags would ask for a term to be added to a loss that is no
        longer there, and the sum would descend fine while meaning nothing: the run would report both
        switches landed and no number afterwards would say which objective it actually was.
        """
        options = self.options
        conflicts = {
            "--rank-weight": options.rank_weight,
            "--distill-weight": options.distill_weight,
            "--dissent-weight": options.dissent_weight,
            "--contrast-weight": options.contrast_weight,
            "--pair-memory": options.pair_memory,
        }
        named = [f"{flag} {value}" for flag, value in conflicts.items() if value]
        if named:
            message = (
                f"--auc-margin {options.auc_margin} is not composed with {', '.join(named)}: it REPLACES "
                "the cross-entropy term those all add themselves to, so what ran would be neither the "
                "margin objective nor the ranking one"
            )
            raise ValueError(message)

    def _check_dissent_alone(self) -> None:
        """The three flags `--dissent-weight` cannot be composed with, each LOUD for `objective`'s reason.

        `--distill-weight` reads the same teacher to pull the member TOWARD the pool's ordering, which is
        the opposite of what this term spends the member on; running both descends a sum whose result
        says nothing about either. `--rank-weight` is the same label ordering at a flat pair mass, and
        `DecorrelatedObjective` deliberately holds it at zero — passing it would be silently ignored,
        which is the exact failure the whole flag vocabulary here exists to prevent. `--pair-memory`
        widens the label term's pool with remembered readouts that carry no teacher, so every remembered
        pair would enter at an emphasis this term cannot compute.
        """
        options = self.options
        conflicts = {
            "--distill-weight": options.distill_weight,
            "--rank-weight": options.rank_weight,
            "--pair-memory": options.pair_memory,
        }
        named = [f"{flag} {value}" for flag, value in conflicts.items() if value]
        if named:
            message = (
                f"--dissent-weight {options.dissent_weight} is not composed with {', '.join(named)}: it "
                "IS the label ranking term at a re-weighted pair mass, and each of those either "
                "descends that same ordering a second time or feeds it pairs with no teacher"
            )
            raise ValueError(message)

    def _check_contrast_alone(self) -> None:
        """The two flags `--contrast-weight` cannot be composed with, each LOUD for `objective`'s reason.

        Not because the terms disagree — a representation term and a pairwise ordering term compose fine,
        and `--rank-weight` is deliberately absent from this list because `ContrastiveObjective` extends
        `RankedObjective` and carries it — but because the DISPATCH here is a single chain returning one
        class, and each of these two returns a different subclass that this branch would silently lose.
        A run passing both would train the other term's recipe under this arm's name and report a number
        about the wrong mechanism, which is the failure every guard in this file exists to prevent.

        `--pair-memory` WAS on this list and is not any more, and it was never a rejected idea: the pool
        is the named fix for this term's small-batch weakness — `contrast_dense` pre-registered exactly
        that suspect against its own -0.0087 — and it was refused only because the chain returned before
        reaching `RememberedRanking`. `contrastive` now dispatches on the pair, so the composition the
        ledger asked for is the one that runs.
        """
        options = self.options
        conflicts = {
            "--distill-weight": options.distill_weight,
            "--dissent-weight": options.dissent_weight,
        }
        named = [f"{flag} {value}" for flag, value in conflicts.items() if value]
        if named:
            message = (
                f"--contrast-weight {options.contrast_weight} is not composed with {', '.join(named)}: "
                "the objective chain returns ONE class, so this branch would win and the term those "
                "flags asked for would never be built"
            )
            raise ValueError(message)

    def reports(self, identifiers: list[str]) -> ReportSupervision | None:
        """The report-embedding supervision this run asked for, or None — and never half of it.

        A weight with no vectors used to be a silent no-op: `if self.report_weight and
        self.report_vectors` skipped the attachment, and the run trained the plain recipe while its
        ledger row, its config and its name all said report supervision. The arm would have been read
        as a refutation of the mechanism rather than of the invocation.
        """
        options = self.options
        if not options.report_weight:
            return None
        if options.report_vectors is None:
            message = (
                f"--report-weight {options.report_weight} asks for report supervision and no "
                "--report-vectors was given, so the run would train the plain recipe under that name"
            )
            raise ValueError(message)
        return ReportSupervision.from_file(options.report_vectors, identifiers, options.report_weight)

    def teacher(self, identifiers: list[str], training: NDArray[np.int64]) -> TeacherTargets | None:
        """The ensemble readings this run asked for, or None — and never half of it, for `reports`' reason.

        Both halves raise, not just the missing file: a `--teacher-targets` with no `--distill-weight`
        loads an array nothing reads, and the arm would report the plain recipe under a distilling name
        just as surely as the other way round. `DistilledObjective` raises on an absent teacher too, but
        it raises on the first BATCH — this raises before the card is touched.
        """
        options = self.options
        # EITHER consumer of the file, because the halves that must arrive together are "a term that
        # reads a teacher" and "a teacher", not "distillation" specifically.
        wants_teacher = options.distill_weight or options.dissent_weight
        if not wants_teacher and options.teacher_targets is None:
            return None
        if options.teacher_targets is None or not wants_teacher:
            message = (
                f"--distill-weight {options.distill_weight} / --dissent-weight {options.dissent_weight} "
                f"and --teacher-targets {options.teacher_targets} are two halves of one mechanism, and "
                "only one was given"
            )
            raise ValueError(message)
        return TeacherTargets.from_file(options.teacher_targets, identifiers, training)

    def model(self, encoder: Encoder, slots: int, design: StudyDesign, device: torch.device) -> StudyModel:
        """The model this run asked for, with every wrapper applied IN ORDER.

        THE ORDER IS THE WHOLE VALUE OF THIS METHOD. Adaptation first, because checkpointing and compile
        both WRAP what they are handed — adapters injected after compilation would sit outside the
        compiled graph, and the run would report a speed and a score belonging to two different models.

        THE DEVICE IS HERE ONLY FOR COMPILATION, and it is not a hedge: inductor's whole gain here is
        fused CUDA kernels, and asking for it on CPU pays a graph build for nothing. On this machine it
        is worse than nothing — dynamo's own `has_triton()` probe indexes a CUDA worker cache that a
        CPU-tensor trace never populated and raises `IndexError: list index out of range`, so a CPU
        forward under compile does not merely go slow, it dies. The model itself is still moved by
        `Trainer`, not here.
        """
        options = self.options
        backbone = encoder.load()
        if options.encoder_weights is not None:
            # BEFORE every wrapper, and that is the same argument the rest of this method makes: weights
            # written into a compiled or adapted backbone land under prefixed keys that the stock
            # reference does not name, which is how a load that matched nothing once reported success.
            backbone = PretrainedEncoder(options.encoder_weights).into(
                backbone, options.variant, options.encoder_adapt_depth
            )
        model = StudyModel(backbone, encoder.dim, slots, design, encoder.normalization)
        if options.lora_rank:
            logger.info(
                "low-rank adaptation: %d layers at rank %d", model.low_rank(options.lora_rank), options.lora_rank
            )
        if options.recompute_activations:
            model.recompute_activations()
        if options.compile_encoder and device.type == "cuda":
            model.compile_encoder()
        return model

    def trainer(
        self,
        model: StudyModel,
        device: torch.device,
        identifiers: list[str],
        training: NDArray[np.int64],
    ) -> Trainer:
        """A `Trainer` with every optional policy this run asked for already attached.

        The policies are attached rather than passed because each is something a run may or may not want,
        and they were accumulating in `main` as a run of `if self.x: trainer.y()` — eight of them by
        the time resampling arrived. The run's SETUP and the run's LOOP are two jobs, and only one of
        them grows every time a flag is added.
        """
        options = self.options
        trainer = Trainer(
            model,
            device,
            augment_config=AugmentConfig() if options.intensity_aug else AugmentConfig.geometric_only(),
            adaptation=self.adaptation(),
            objective=self.objective(),
        )
        if options.bf16:
            trainer.bfloat16()
        if options.whole_stack:
            trainer.whole_stack(options.slices)
        if options.train_size:
            trainer.resample_to(options.train_size)
        reports = self.reports(identifiers)
        if reports is not None:
            trainer.report_supervision(reports)
        teacher = self.teacher(identifiers, training)
        if teacher is not None:
            trainer.teach_from(teacher)
        if options.ema:
            trainer.exponential_average()
        # BEFORE the schedule, which is not cosmetic: the schedule's total is counted in OPTIMISER steps
        # and the trainer's own accumulation is what converts micro-batches into them.
        trainer.accumulate(options.accumulate)
        trainer.blend(options.mixup)
        if options.one_cycle:
            # FLOOR now, and it was ceil for a real reason that no longer holds: the batcher used to
            # KEEP the final partial batch, so a floor undercounted by one step per epoch and OneCycleLR
            # raised on the very last step of the run — after every epoch had been logged and before the
            # checkpoint was written, the most expensive place to fail. Training now drops that batch so
            # the encoder's graph has ONE shape, and the step count must follow it exactly.
            # And in OPTIMISER steps, not micro-batches: under accumulation the loop steps once per
            # group, so counting micro-batches would size the cycle `accumulate` times too long and the
            # run would end mid-warmup-and-decay having never reached the low rate the schedule exists
            # to finish at. Same class of failure as the ceil above, opposite sign.
            micro_batches = max(len(training) // options.batch, 1)
            steps_per_epoch = trainer.accumulation.groups(micro_batches)
            trainer.one_cycle(options.epochs * max(steps_per_epoch, 1))
        return trainer
