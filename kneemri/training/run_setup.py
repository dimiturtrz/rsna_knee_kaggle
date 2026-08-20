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

import torch
from transformers import AutoModel

from kneemri.models.backbone import Encoder
from kneemri.models.model import StudyDesign, StudyModel
from kneemri.training.augmentation import AugmentConfig
from kneemri.training.landed import Request
from kneemri.training.objective import Objective, RankedObjective
from kneemri.training.options import TrainingOptions
from kneemri.training.pretrained_encoder import PretrainedEncoder
from kneemri.training.remembered_ranking import RememberedRanking
from kneemri.training.report_supervision import ReportSupervision
from kneemri.training.trainer import Adaptation, Trainer

logger = logging.getLogger(__name__)


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
            recompute=options.recompute_activations,
            compiling=options.compile_encoder and device.type == "cuda",
            bf16=options.bf16,
            whole_stack=options.whole_stack,
            slices=options.slices,
            train_size=options.train_size,
            ema=options.ema,
            one_cycle=options.one_cycle,
            reports=options.report_vectors is not None,
            rank_weight=options.rank_weight,
            pair_memory=options.pair_memory,
            per_label_loss=options.per_label_loss,
            normalise_by_count=options.normalise_by_count,
            intensity_aug=options.intensity_aug,
            weight_floor=options.weight_floor,
            calibrated_targets=options.calibrated_targets,
            severity_weight=options.severity_weight,
            unfreeze_last=options.unfreeze_last,
        )

    def objective(self) -> Objective:
        """The loss policy these flags ask for.

        Off the command line rather than inside `main` so the choice can be checked without a GPU, a
        corpus or a downloaded encoder. The alternative is finding out that a queued arm was miswired
        after it has already waited forty minutes for the card.
        """
        options = self.options
        if options.pair_memory and not options.rank_weight:
            # LOUD, for the reason `reports` gives below: a memory with no ranking term to feed is a
            # silent no-op that trains the plain recipe under the arm's name, and would be read as a
            # refutation of the mechanism rather than of the invocation.
            message = (
                f"--pair-memory {options.pair_memory} widens the ranking term's pair pool and no "
                "--rank-weight was given, so there is no ranking term for it to widen"
            )
            raise ValueError(message)
        if options.pair_memory:
            return RememberedRanking(
                options.rank_weight,
                capacity=options.pair_memory,
                normalise_by_count=options.normalise_by_count,
                per_label=options.per_label_loss,
            )
        if options.rank_weight:
            return RankedObjective(
                options.rank_weight,
                normalise_by_count=options.normalise_by_count,
                per_label=options.per_label_loss,
            )
        return Objective(normalise_by_count=options.normalise_by_count, per_label=options.per_label_loss)

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
        backbone = AutoModel.from_pretrained(encoder.repository)
        if options.encoder_weights is not None:
            # BEFORE every wrapper, and that is the same argument the rest of this method makes: weights
            # written into a compiled or adapted backbone land under prefixed keys that the stock
            # reference does not name, which is how a load that matched nothing once reported success.
            backbone = PretrainedEncoder(options.encoder_weights).into(backbone, options.variant)
        model = StudyModel(backbone, encoder.dim, slots, design)
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
        training_positions: int,
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
        if options.ema:
            trainer.exponential_average()
        if options.one_cycle:
            # FLOOR now, and it was ceil for a real reason that no longer holds: the batcher used to
            # KEEP the final partial batch, so a floor undercounted by one step per epoch and OneCycleLR
            # raised on the very last step of the run — after every epoch had been logged and before the
            # checkpoint was written, the most expensive place to fail. Training now drops that batch so
            # the encoder's graph has ONE shape, and the step count must follow it exactly.
            steps_per_epoch = max(training_positions // options.batch, 1)
            trainer.one_cycle(options.epochs * max(steps_per_epoch, 1))
        return trainer
