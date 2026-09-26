"""What a run's flags BUILD, separated from what they say.

`TrainConfig` answered two different questions and was measured doing it: LCOM4 = 5, the lowest
cohesion of any class here, and the one cluster that never touched the others was this — the seven
methods that turn flags into the objects a fit runs on. The other clusters read a flag pair as a value
(`crop`, `band`, `reading`) or read the corpus (`cache`, `sampled`); none of them constructs anything.
"""

import logging
from dataclasses import dataclass, replace

import numpy as np
import torch
from numpy.typing import NDArray

from kneemri.models.backbone import ENCODERS
from kneemri.models.encoder import Encoder
from kneemri.models.model import StudyDesign, StudyModel
from kneemri.training.adaptation import Adaptation
from kneemri.training.augmentation import AugmentConfig
from kneemri.training.config_tree import TrainConfig
from kneemri.training.device_budget import DeviceBudget
from kneemri.training.feature_distillation import FeatureDistillation, Widths
from kneemri.training.grades import Grades, OrdinalGrades
from kneemri.training.label_focus import LabelFocus
from kneemri.training.landed import Request
from kneemri.training.objective import LabelStructure, Objective, Shaping
from kneemri.training.objective_checks import ObjectiveChecks
from kneemri.training.patch_erasing import PatchErasing
from kneemri.training.pretrained_encoder import PretrainedEncoder
from kneemri.training.progress_schedule import ProgressSchedule
from kneemri.training.remembered import Remembered
from kneemri.training.report_supervision import ReportSupervision
from kneemri.training.sampling import Disagreed, MixedCorpus, Ramped, Sampling, Uniform
from kneemri.training.side_ranking import PairedSides, SideRanking
from kneemri.training.teacher_features import TeacherFeatures
from kneemri.training.teacher_targets import TeacherTargets
from kneemri.training.terms import (
    CONTRAST_TERM,
    DISSENT_TERM,
    DISTILL_TERM,
    FEATURE_TERM,
    MARGIN_TERM,
    Contrast,
    Dissent,
    Distillation,
    Margin,
    Pooled,
    Ranking,
    Term,
)
from kneemri.training.trainer import Trainer

logger = logging.getLogger(__name__)

ENCODER_FULLY_OPEN = 1_000
"""More blocks than any trunk here runs, so `min` with the encoder's depth opens all of them.

Under LoRA `unfreeze_last` describes BASE weights, which are frozen, while the adapters it says nothing
about sit in every block and retain their activations. So the honest trainable count is the whole trunk,
and this is how the clamp says that without naming a depth that would go stale."""

HARD_AND_RAMP = (
    "--hard-* and --certainty-ramp/--sample-ramp both schedule example hardness; the integer ramps are "
    "kept only to reproduce recorded arms, so pick one"
)


@dataclass(frozen=True)
class RunSetup:
    """The model, the trainer and the policies one run's flags ask for.

    Everything here is checkable without a card, a corpus or a downloaded encoder, which is the point:
    a miswired switch that surfaces after forty minutes of queue costs the slot, and every one of these
    methods exists because a run once trained the plain recipe under an arm's name.
    """

    options: TrainConfig

    @classmethod
    def of(cls, argv: list[str]) -> "RunSetup":
        """The setup a command line asks for — what the queue's memory gate and the smoke both price."""
        return cls(TrainConfig.of(argv))

    def adaptation(self) -> Adaptation:
        """How the encoder adapts: which blocks move and how fast, which are one decision."""
        return Adaptation(
            head_lr=self.options.encoder.head_lr,
            backbone_lr=self.options.encoder.backbone_lr,
            unfreeze_last=self.options.encoder.unfreeze_last,
            unfreeze_offset=self.options.encoder.unfreeze_offset,
            unfreeze_stem=self.options.encoder.unfreeze_stem,
            layer_decay=self.options.encoder.layer_decay,
            weight_decay=self.options.encoder.weight_decay,
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
        encoder = ENCODERS[self.options.encoder.variant]
        size = self.options.data.train_size or self.options.data.image_size
        opened = ENCODER_FULLY_OPEN if self.options.encoder.lora_rank else self.options.encoder.unfreeze_last
        return DeviceBudget(
            self.options.run.vram_gb,
            self.options.data.slices,
            encoder.activation_units(size),
            self.options.optim.batch,
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
            lora_rank=options.encoder.lora_rank,
            slice_regions=options.model.slice_regions,
            recompute=options.encoder.recompute_activations,
            compiling=options.run.compile_encoder and device.type == "cuda",
            bf16=options.run.bf16,
            whole_stack=options.encoder.whole_stack,
            slices=options.data.slices,
            train_size=options.data.train_size,
            ema=options.optim.ema,
            one_cycle=options.optim.one_cycle,
            accumulate=options.optim.accumulate,
            reports=options.run.report_vectors is not None,
            rank_weight=options.targets.rank_weight,
            distill_weight=options.teacher.distill_weight,
            feature_weight=options.teacher.feature_weight,
            dissent_weight=options.teacher.dissent_weight,
            contrast_weight=options.targets.contrast_weight,
            auc_margin=options.targets.auc_margin,
            teaching=options.teacher.teacher_targets is not None,
            teaching_features=options.teacher.teacher_features is not None,
            pair_memory=options.targets.pair_memory,
            per_label_loss=options.targets.per_label_loss,
            normalise_by_count=options.targets.normalise_by_count,
            intensity_aug=options.regularisation.intensity_aug,
            weight_floor=options.targets.weight_floor,
            calibrated_targets=options.targets.calibrated_targets,
            finding_calibration=options.targets.finding_calibration,
            severity_weight=options.targets.severity_weight,
            hold_annotations=options.targets.hold_annotations,
            hold_annotation_studies=options.targets.hold_annotation_studies,
            grade_ladder=options.targets.grade_ladder,
            # The path as a STRING, and the empty string for the null case, because `Targets.supervised`
            # reports it that way and this comparison is a strict equality against exactly that report.
            label_errors=str(options.targets.label_errors) if options.targets.label_errors else "",
            unread_targets=str(options.targets.unread_targets) if options.targets.unread_targets else "",
            side_weight=options.targets.side_weight,
            # The EMPTY STRING and not None, because that is what `LabelFocus.settings()` reports for
            # the null policy, and this comparison is a strict equality against exactly that report.
            focus_label=options.targets.focus_label or "",
            unfocused_weight=options.targets.unfocused_weight,
            unfreeze_last=options.encoder.unfreeze_last,
            unfreeze_offset=options.encoder.unfreeze_offset,
            unfreeze_stem=options.encoder.unfreeze_stem,
            layer_decay=options.encoder.layer_decay,
            mixup=options.regularisation.mixup,
            weight_decay=options.encoder.weight_decay,
            warmup=options.optim.warmup,
            grad_clip=options.optim.grad_clip,
            sharpness=options.optim.sharpness,
            stochastic_depth=options.regularisation.stochastic_depth,
            backbone_lr=options.encoder.backbone_lr,
            sample_spread=options.sampling.sample_spread,
            sample_alpha=options.sampling.sample_alpha,
            certainty_ramp=options.optim.certainty_ramp,
            sample_ramp=options.sampling.sample_ramp,
            aug_start=options.regularisation.aug_start,
            aug_mid=options.regularisation.aug_mid,
            aug_end=options.regularisation.aug_end,
            aug_ramp=options.regularisation.aug_ramp,
            aug_tail=options.regularisation.aug_tail,
            hard_start=options.sampling.hard_start,
            hard_mid=options.sampling.hard_mid,
            hard_end=options.sampling.hard_end,
            hard_ramp=options.sampling.hard_ramp,
            hard_tail=options.sampling.hard_tail,
            corpus_oversample=options.sampling.corpus_oversample,
            target_corpus=options.sampling.target_corpus,
            slot_dropout=options.regularisation.slot_dropout,
            patch_erase=options.regularisation.patch_erase,
            ghosting=options.regularisation.ghosting,
            scale_jitter=options.regularisation.scale_jitter,
            padding_mode=options.regularisation.padding_mode,
        )

    def erasing(self, identifiers: list[str]) -> PatchErasing | None:
        """The occlusion maps this run erases from, joined to its own study order, or None for no erasing.

        A rate with no maps has nothing to price a cell by, and erasing nothing IS the control: such a run
        would train the unerased recipe and report it under the erased arm's name, so it is refused rather
        than quietly zeroed.
        """
        options = self.options
        if options.regularisation.patch_maps is not None:
            return PatchErasing.loaded(
                options.regularisation.patch_maps, identifiers, options.regularisation.patch_erase
            )
        if options.regularisation.patch_erase > 0.0:
            message = (
                f"--patch-erase {options.regularisation.patch_erase} needs --patch-maps; drop the rate or name the maps"
            )
            raise ValueError(message)
        return None

    def hardness(self) -> ProgressSchedule:
        """The example-hardness schedule these flags ask for, refusing one beside a legacy integer ramp.

        Two schedules over the same weights would multiply, and the recorded ramps would stop reproducing.
        """
        options = self.options
        schedule = ProgressSchedule(
            options.sampling.hard_start,
            options.sampling.hard_mid,
            options.sampling.hard_end,
            options.sampling.hard_ramp,
            options.sampling.hard_tail,
        )
        if schedule != ProgressSchedule() and (options.optim.certainty_ramp or options.sampling.sample_ramp):
            raise ValueError(HARD_AND_RAMP)
        return schedule

    def drawing(self, identifiers: list[str], corpus: NDArray[np.str_] | None = None) -> Sampling:
        """The draw policy these flags ask for, off the command line and without a card.

        `--sample-ramp` is a modifier on `--sample-alpha` rather than a policy of its own: a ramp to a
        tilt of zero is the alpha-zero control, and a ramp of zero epochs is the immediate tilt. Both
        degenerate cases resolve HERE rather than inside the sampler, so the two recipes already in the
        ledger parse to exactly the objects they ran under.

        `--corpus-oversample` is the Scheme B draw: it weights each position by whether its `corpus`
        tag matches `--target-corpus`. It cannot be combined with `--sample-spread`, because both are
        epoch-level draw policies.
        """
        options = self.options
        if options.sampling.corpus_oversample > 0.0 and options.sampling.sample_spread is not None:
            message = (
                f"--corpus-oversample {options.sampling.corpus_oversample} and --sample-spread cannot both steer "
                "the epoch draw; pick one sampling policy"
            )
            raise ValueError(message)
        if options.sampling.corpus_oversample > 0.0:
            if corpus is None:
                message = (
                    f"--corpus-oversample {options.sampling.corpus_oversample} needs a `corpus` column in the "
                    "labels table, which this one does not carry"
                )
                raise ValueError(message)
            return MixedCorpus(corpus, options.sampling.target_corpus, options.sampling.corpus_oversample)
        if options.sampling.sample_spread is None:
            return Uniform()
        if options.sampling.sample_ramp > 0:
            return Ramped.loaded(
                options.sampling.sample_spread, identifiers, options.sampling.sample_alpha, options.sampling.sample_ramp
            )
        return Disagreed.loaded(options.sampling.sample_spread, identifiers, options.sampling.sample_alpha)

    def objective(self, widths: Widths | None = None) -> Objective:
        """The loss policy these flags ask for.

        Off the command line rather than inside `main` so the choice can be checked without a GPU, a
        corpus or a downloaded encoder. The alternative is finding out that a queued arm was miswired
        after it has already waited forty minutes for the card.

        `widths` is the ONE thing the flags cannot say: a feature term projects the student's context
        onto the teacher's width, and neither number is on the command line — one is a fact about the
        built model and the other about the loaded npz. It defaults to None so every caller that only
        wants to know WHICH objective a recipe asks for keeps working without either.
        """
        options = self.options
        ObjectiveChecks(options).composed()
        structure = LabelStructure(
            grades=OrdinalGrades() if options.targets.grade_ladder else Grades(),
            sides=PairedSides(options.targets.side_weight) if options.targets.side_weight else SideRanking(),
            focus=LabelFocus.parse(options.targets.focus_label, options.targets.unfocused_weight),
        )
        shaping = Shaping(
            normalise_by_count=options.targets.normalise_by_count,
            per_label=options.targets.per_label_loss,
            structure=structure,
        )
        terms = self.terms(widths)
        if options.targets.auc_margin:
            return Objective(base=terms[-1], terms=terms[:-1], shaping=shaping)
        return Objective(terms=terms, shaping=shaping)

    def terms(self, widths: Widths | None = None) -> list[Term]:
        """The terms these flags ask for: the ranking term if weighted, then at most ONE exclusive term.

        REFUSES two exclusive terms rather than picking one. The chain this replaced checked margin,
        dissent, distill, feature and contrast in that order and returned the first it found, so an arm
        asking for two trained one under both names. `ObjectiveChecks.composed` names each pair it has
        a reason for; this is the backstop for a pair nobody has argued yet.

        `rank_weight` is carried beside whichever exclusive term is asked, for `Distillation`'s reason:
        the house recipe passes --rank-weight 1.0, so excluding it would make every such arm a
        two-variable change against the control it is measured on. The remembered pool widens the LAST
        term — the exclusive one when there is one, the ranking term otherwise.
        """
        options = self.options
        exclusive = {
            MARGIN_TERM: (options.targets.auc_margin, lambda: Margin(options.targets.auc_margin)),
            DISSENT_TERM: (options.teacher.dissent_weight, lambda: Dissent(options.teacher.dissent_weight)),
            DISTILL_TERM: (options.teacher.distill_weight, lambda: Distillation(options.teacher.distill_weight)),
            FEATURE_TERM: (options.teacher.feature_weight, lambda: self._features(widths)),
            CONTRAST_TERM: (options.targets.contrast_weight, lambda: Contrast(options.targets.contrast_weight)),
        }
        asked = [key for key, (value, _) in exclusive.items() if value]
        if len(asked) > 1:
            message = (
                f"{' and '.join(asked)} are each an exclusive loss term and this run asks for {len(asked)} of "
                "them — the objective builder used to keep the first and silently drop the rest"
            )
            raise ValueError(message)
        parts: list[Term] = [Ranking(options.targets.rank_weight)] if options.targets.rank_weight else []
        parts += [exclusive[key][1]() for key in asked]
        if options.targets.pair_memory:
            parts[-1] = Remembered(self._pooled(parts[-1]), options.targets.pair_memory)
        return parts

    def _pooled(self, term: Term) -> Pooled:
        """`term` as one a remembered pool can widen, or a refusal naming it."""
        if not isinstance(term, Pooled):
            message = f"--pair-memory {self.options.targets.pair_memory} cannot widen {term.key}, which pools nothing"
            raise ValueError(message)
        return term

    def _features(self, widths: Widths | None) -> FeatureDistillation:
        """The feature term, built against the two widths it projects between.

        RAISES when the widths are absent rather than assuming they are equal. A caller asking which
        objective a recipe wants — `arm_check`, `vram_need` — never reaches here because the flag it
        dispatches on is off; a caller that reached here with no widths is one that will fit, and
        guessing on its behalf would build a projection between two numbers nobody measured.
        """
        if widths is None:
            message = (
                f"--feature-weight {self.options.teacher.feature_weight} builds a projection from the student's "
                "context width to the teacher's, and neither was supplied — this call cannot know "
                "either number, so it must not guess one"
            )
            raise ValueError(message)
        return FeatureDistillation(widths, self.options.teacher.feature_weight)

    def reports(self, identifiers: list[str]) -> ReportSupervision | None:
        """The report-embedding supervision this run asked for, or None — and never half of it.

        A weight with no vectors used to be a silent no-op: `if self.report_weight and
        self.report_vectors` skipped the attachment, and the run trained the plain recipe while its
        ledger row, its config and its name all said report supervision. The arm would have been read
        as a refutation of the mechanism rather than of the invocation.
        """
        options = self.options
        if not options.encoder.report_weight:
            return None
        if options.run.report_vectors is None:
            message = (
                f"--report-weight {options.encoder.report_weight} asks for report supervision and no "
                "--report-vectors was given, so the run would train the plain recipe under that name"
            )
            raise ValueError(message)
        return ReportSupervision.from_file(options.run.report_vectors, identifiers, options.encoder.report_weight)

    def teacher(self, identifiers: list[str], training: NDArray[np.int64]) -> TeacherTargets | None:
        """The ensemble readings this run asked for, or None — and never half of it, for `reports`' reason.

        Both halves raise, not just the missing file: a `--teacher-targets` with no `--distill-weight`
        loads an array nothing reads, and the arm would report the plain recipe under a distilling name
        just as surely as the other way round. `Distillation` raises on an absent teacher too, but
        it raises on the first BATCH — this raises before the card is touched.
        """
        options = self.options
        # EITHER consumer of the file, because the halves that must arrive together are "a term that
        # reads a teacher" and "a teacher", not "distillation" specifically.
        wants_teacher = options.teacher.distill_weight or options.teacher.dissent_weight
        if not wants_teacher and options.teacher.teacher_targets is None:
            return None
        if options.teacher.teacher_targets is None or not wants_teacher:
            message = (
                f"--distill-weight {options.teacher.distill_weight} "
                f"/ --dissent-weight {options.teacher.dissent_weight} "
                f"and --teacher-targets {options.teacher.teacher_targets} are two halves of one mechanism, and "
                "only one was given"
            )
            raise ValueError(message)
        return TeacherTargets.from_file(options.teacher.teacher_targets, identifiers, training)

    def teacher_embeddings(self, identifiers: list[str], training: NDArray[np.int64]) -> TeacherFeatures | None:
        """The teacher REPRESENTATIONS this run asked for, or None — never half, for `teacher`'s reason.

        A `--teacher-features` with no `--feature-weight` loads an array nothing reads, and a weight with
        no file would descend the plain recipe under a feature arm's name; both raise here, before the
        card is touched, rather than on the first batch.
        """
        options = self.options
        if not options.teacher.feature_weight and options.teacher.teacher_features is None:
            return None
        if options.teacher.teacher_features is None or not options.teacher.feature_weight:
            message = (
                f"--feature-weight {options.teacher.feature_weight} and --teacher-features "
                f"{options.teacher.teacher_features} are two halves of one mechanism, and only one was given"
            )
            raise ValueError(message)
        return TeacherFeatures.from_file(options.teacher.teacher_features, identifiers, training)

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
        if options.encoder.encoder_weights is not None:
            # BEFORE every wrapper, and that is the same argument the rest of this method makes: weights
            # written into a compiled or adapted backbone land under prefixed keys that the stock
            # reference does not name, which is how a load that matched nothing once reported success.
            backbone = PretrainedEncoder(options.encoder.encoder_weights).into(
                backbone, options.encoder.variant, options.encoder.encoder_adapt_depth
            )
        model = StudyModel(backbone, encoder.dim, slots, design, encoder.normalization)
        if options.encoder.lora_rank:
            logger.info(
                "low-rank adaptation: %d layers at rank %d",
                model.low_rank(options.encoder.lora_rank),
                options.encoder.lora_rank,
            )
        if options.encoder.recompute_activations:
            model.recompute_activations()
        if options.run.compile_encoder and device.type == "cuda":
            model.compile_encoder()
        return model

    def trainer(
        self,
        model: StudyModel,
        device: torch.device,
        identifiers: list[str],
        training: NDArray[np.int64],
        corpus: NDArray[np.str_] | None = None,
    ) -> Trainer:
        """A `Trainer` with every optional policy this run asked for already attached.

        The policies are attached rather than passed because each is something a run may or may not want,
        and they were accumulating in `main` as a run of `if self.x: trainer.y()` — eight of them by
        the time resampling arrived. The run's SETUP and the run's LOOP are two jobs, and only one of
        them grows every time a flag is added.
        """
        options = self.options
        # LOADED BEFORE THE TRAINER, which is the one ordering constraint in this method: the objective's
        # projection is sized from the teacher's width, the optimiser is assembled from the objective at
        # construction, and neither can be revised afterwards.
        features = self.teacher_embeddings(identifiers, training)
        trainer = Trainer(
            model,
            device,
            # `replace` rather than a keyword on each branch: slot dropout and field-of-view jitter are
            # both orthogonal to the intensity block, so spelling either twice would let the two recipes
            # drift by one edit. `geometric_only` carries the same 0.08 scale as the full config, so the
            # override is the identity at the default and every ledger number keeps its meaning.
            augment_config=replace(
                AugmentConfig() if options.regularisation.intensity_aug else AugmentConfig.geometric_only(),
                slot_dropout=options.regularisation.slot_dropout,
                scale_jitter=options.regularisation.scale_jitter,
                padding_mode=options.regularisation.padding_mode,
                ghosting=options.regularisation.ghosting,
            ),
            adaptation=self.adaptation(),
            objective=self.objective(None if features is None else Widths(model.context_width(), features.width())),
        )
        if options.run.bf16:
            trainer.bfloat16()
        if options.encoder.whole_stack:
            trainer.whole_stack(options.data.slices)
        if options.data.train_size:
            trainer.resample_to(options.data.train_size)
        reports = self.reports(identifiers)
        if reports is not None:
            trainer.report_supervision(reports)
        teacher = self.teacher(identifiers, training)
        if teacher is not None:
            trainer.teach_from(teacher)
        if features is not None:
            trainer.teach_features_from(features)
        if options.optim.ema:
            trainer.exponential_average()
        erasing = self.erasing(identifiers)
        if erasing is not None:
            trainer.erase_with(erasing)
        # BEFORE the schedule, which is not cosmetic: the schedule's total is counted in OPTIMISER steps
        # and the trainer's own accumulation is what converts micro-batches into them.
        trainer.accumulate(options.optim.accumulate)
        trainer.blend(options.regularisation.mixup)
        trainer.clip_gradients(options.optim.grad_clip)
        # AFTER `accumulate` above, and that order is forced: `sharpen` refuses a radius under a group
        # larger than one micro-batch, and it can only see the group if the group is already attached.
        trainer.sharpen(options.optim.sharpness)
        # BEFORE the optimiser groups are read by anything downstream is not a constraint here: stochastic
        # depth registers no parameter, so the groups built in the constructor still name every tensor.
        trainer.drop_paths(options.regularisation.stochastic_depth)
        # ATTACHED UNCONDITIONALLY, unlike the policies above it, because the null here is an object and
        # not an absence — and because a policy only some runs attach is a policy the options-to-policies
        # walk cannot see. ALIGNED AGAINST `identifiers`, the corpus's own study order, and not against
        # `training`: the policy is asked for a POSITION's weight, and positions index the corpus.
        trainer.draw_with(self.drawing(identifiers, corpus))
        # BEFORE the schedule too, and for a sharper reason than accumulation's: `one_cycle` reads the
        # warmup fraction when it BUILDS OneCycleLR, so a warm_up after it would set a field nothing
        # reads again and the run would anneal on 0.15 while its config recorded the value asked for.
        trainer.warm_up(options.optim.warmup)
        trainer.ramp_certainty(options.optim.certainty_ramp)
        trainer.pace(
            options.optim.epochs,
            strength=ProgressSchedule(
                options.regularisation.aug_start,
                options.regularisation.aug_mid,
                options.regularisation.aug_end,
                options.regularisation.aug_ramp,
                options.regularisation.aug_tail,
            ),
            hardness=self.hardness(),
        )
        if options.optim.one_cycle:
            # FLOOR now, and it was ceil for a real reason that no longer holds: the batcher used to
            # KEEP the final partial batch, so a floor undercounted by one step per epoch and OneCycleLR
            # raised on the very last step of the run — after every epoch had been logged and before the
            # checkpoint was written, the most expensive place to fail. Training now drops that batch so
            # the encoder's graph has ONE shape, and the step count must follow it exactly.
            # And in OPTIMISER steps, not micro-batches: under accumulation the loop steps once per
            # group, so counting micro-batches would size the cycle `accumulate` times too long and the
            # run would end mid-warmup-and-decay having never reached the low rate the schedule exists
            # to finish at. Same class of failure as the ceil above, opposite sign.
            micro_batches = max(len(training) // options.optim.batch, 1)
            steps_per_epoch = trainer.accumulation.groups(micro_batches)
            trainer.one_cycle(options.optim.epochs * max(steps_per_epoch, 1))
        return trainer
