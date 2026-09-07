"""Did the switches this run asked for actually reach the model? Checked before the card is spent.

EVERY OPTIONAL POLICY IS APPLIED BY A CALL THAT CAN SILENTLY DO NOTHING, and this project has already
paid for three of them. `low_rank` chose its layers from a NAME LIST once, so on the encoder whose
projections were named differently it wrapped zero layers and the run trained a plain model under a
LoRA arm's name. A `str.replace` that matched nothing left an arm writing one checkpoint name for every
read size. The trainable-parameter count that was supposed to validate the first of those counted
parameters that were never adapters, so a wrecking ball read as the expected 6.4%.

The shape of that failure is always the same: the flag parses, the config records it, the log says the
run wanted it, and nothing asks the model whether it happened. Forty minutes of card later the result
is a member trained under a recipe nobody ran.

WHAT THIS COMPARES are two dicts in one vocabulary: what the run asked for, and what the objects say
they became. NEITHER SIDE IS READ HERE — the model reports its own `applied()` and the trainer its own
`policies()`, because each is the authority on itself and a reader that reached into both would be
this module deciding what checkpointing means. This file owns only the COMPARISON and the translation
from flags into that vocabulary.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from core.data.corpus.slot_reader import GROUP
from core.data.labels.targets import (
    ANNOTATIONS,
    CALIBRATED,
    CONFIDENCE_FLOOR,
    FINDING,
    SEVERITY_WEIGHT,
    SUSPECT,
    UNREAD,
    WEIGHT_FLOOR,
    Targets,
)
from kneemri.models.encoder_surgery import NO_OFFSET
from kneemri.models.model import (
    ADAPTERS,
    COMPILED_ENCODER,
    DEPTH_REGIONS,
    FROZEN_TOP,
    OPENED_BLOCKS,
    RECOMPUTING,
    StudyModel,
)
from kneemri.training.augmentation import INTENSITY_AUGMENT, SLOT_DROPOUT
from kneemri.training.grades import GRADE_LADDER
from kneemri.training.label_focus import FOCUS, UNFOCUSED
from kneemri.training.objective import (
    CONTRAST_TERM,
    COUNT_NORMALISED,
    DISSENT_TERM,
    DISTILL_TERM,
    FEATURE_TERM,
    MARGIN_TERM,
    PAIR_POOL,
    PER_LABEL,
    RANK_TERM,
)
from kneemri.training.side_ranking import SIDE_TERM
from kneemri.training.teacher_features import FEATURE_TEACHER
from kneemri.training.teacher_targets import TEACHER
from kneemri.training.trainer import (
    ACCUMULATION,
    AUTOCAST,
    AVERAGING,
    BACKBONE_RATE,
    CERTAINTY_RAMP,
    CLIPPING,
    CORPUS_OVERSAMPLE,
    LAYER_DECAY,
    MIXUP,
    OPTIMISER_DECAY,
    READ_SIZE,
    REPORTS,
    SAMPLING,
    SAMPLING_RAMP,
    SCHEDULED,
    TARGET_CORPUS,
    WARMUP,
    WINDOW,
    Trainer,
)

FLOAT_TOLERANCE = 1e-6
"""How far a derived float may drift from the value asked for and still count as landed. A switch that
did not land misses by its whole magnitude; a switch that did misses by the last bits of a multiply."""

NOT_A_SWITCH = {
    "paths": "the data root, machine-specific and gitignored",
    "labels": "the label table, an input rather than a policy",
    "eval_labels": "the holdout table, same",
    "out": "where the run writes, set per invocation",
    "init_scope": "which of `--init`'s tensors are taken, and the load ANSWERS it: a name outside "
    "`INIT_SCOPES` raises at parse, and every tensor the scope carries is still shape-checked into the "
    "parameter it names. A scope that dropped the wrong thing shows up as an absent key, not as a switch "
    "that quietly did not land",
    "init": "the weights the run BEGINS from, and `WarmStart.applied` loads them STRICTLY: a checkpoint "
    "that does not fit the model this run's flags describe raises there, before the card is spent, and a "
    "checkpoint that does fit is already in the parameters. There is nothing left for an equality here to "
    "catch — and the provenance a landed-check would report is the line `applied` returns, which the run "
    "logs and `member_record` keeps",
    "epochs": "run length, which nothing downstream can silently ignore",
    "studies": "corpus truncation, visible in the split line every run logs",
    "train_fraction": "which TRAINING studies were fitted on — the same kind of thing, and visible in "
    "the same split line. No model, trainer or targets object holds it, so there is nothing here to "
    "compare it against; what a landed-check would want is the study COUNT, and the run logs that",
    "patience": "when to stop, read by the stopper each epoch rather than applied once",
    "batch": "how many studies a step holds, a resource knob the loader obeys or raises on",
    "vram_gb": "a memory cap set by the queue against the card it runs on",
    "seed": "a seed is not a method; folds and seeds are banned as arms here",
    "split_seed": "same, and it moves the split rather than the recipe",
    "folds": "banned for the same reason",
    "fold": "which split half is held out, chosen per invocation",
    "exclude_fold": "which fold is dropped from TRAINING as well as from the holdout — a cut of the "
    "corpus, like `fold`, and the split line every run logs is what shows it landed. No model, trainer "
    "or targets object holds it, so there is nothing here to compare against; the recipe block records "
    "the value and the study count is what changes",
    "image_size": "a cache axis — guarded by the cache FILE NAME, not by a policy object",
    "crop_mm": "a cache axis",
    "centre": "a cache axis",
    "band_low": "a cache axis",
    "band_high": "a cache axis",
    "spacing": "a cache axis",
    "scheme": "a cache axis",
    "tie": "a cache axis",
    "cache_studies": "a cache axis — it is in the name as `_first{N}`",
    "compile_encoder": "resolved into `Request.compiling`, which `main` decides on the device",
    "recompute_activations": "resolved into `Request.recompute`",
    "report_vectors": "resolved into `Request.reports`",
    "teacher_targets": "resolved into `Request.teaching`: the check asks whether readings ARRIVED, and "
    "WHICH file they came from is recorded by `member_record` rather than compared here",
    "teacher_features": "resolved into `Request.teaching_features`, for the reason above and separately "
    "from it: a run may carry one file, the other, both or neither, and one merged switch would answer "
    "'a teacher arrived' for the arm whose teacher did not",
    "report_weight": "resolved into `Request.reports` alongside it",
    "variant": "chosen by CONSTRUCTION: the registry raises on a name it cannot build",
    "encoder_layer": "constructed, and the width check above raises when the registry and weights disagree",
    "pool": "constructed — `POOL_PARTS[self.pool]` raises on a pooling that does not exist",
    "slice_pooling": "constructed — `SLICE_POOLINGS[...]` raises before any weight is fetched",
    "through_plane": "constructed into `StudyDesign` and stamped into the manifest: the encoder's three "
    "input channels hold (i-1, i, i+1) instead of three copies of i, so a rebuild that defaults it feeds "
    "the trunk different pixels and moves the member's own fingerprint",
    "slot_pooling": "constructed — `SLOT_POOLINGS[...]` likewise",
    "anatomical_prior": "constructed, and a slot count the prior does not cover RAISES rather than dropping it",
    "head_lr": "a rate, not a switch: it is the group's `lr` and no path applies it partly",
    # `backbone_lr` was written off here on that same argument until 2026-09-05. It is COMPARED now, as
    # BACKBONE_RATE. The argument was not wrong about the path, which is short and direct — it was wrong
    # that a short path needs no witness: `band_deep_matched` floored its loss 0.189 above its control
    # with every recorded flag correct, and the one reading that would have said whether the rate the
    # depth arm was compensated with ever reached the optimiser did not exist to take (knee-62sol).
    "encoder_weights": "an INITIALISATION that cannot land silently: `PretrainedEncoder.into` raises on a "
    "variant mismatch and on any key the target does not name, so a load that did nothing is an exception "
    "rather than a control's score reported under the arm's name",
    "encoder_adapt_depth": "how much of `encoder_weights` is taken, so it is the same INITIALISATION "
    "written off above rather than a second behaviour — and it is equally loud: the loader reports how "
    "many tensors it left stock, and a depth past the last block simply loads everything",
}
"""Every `TrainingOptions` field this check does NOT compare, each with WHY — a decision record.

The partition is made TOTAL and asserted at the unit gate, because 'no `Request` field names it' is
true both of a knob that cannot land silently and of one nobody has checked yet, and only the second
is a hole. `switch_coverage.NOT_A_RECIPE` does the same for the arms; this is the same forcing move on
the axis that matters at the card: a switch added to the trainer now decides here whether it can fail
silently, rather than being discovered as forty minutes trained under a recipe nobody ran.

Named by their `TrainingOptions` field, but SPELLED HERE rather than imported from it — that dataclass
builds the model and the trainer this module compares, so importing it would close the import cycle
`Request`'s primitive fields exist to keep open.
"""


class Landed:
    """What a built model, trainer and target table BECAME, against what the run asked them to be.

    THE TARGETS ARE HERE because a switch that maps to a DATA argument rather than to a policy object
    was outside every check this project has: `options_to_policies` walks the model and the trainer, so
    the two label switches were the one pair of flags nothing downstream could contradict.
    """

    def __init__(self, model: StudyModel, trainer: Trainer, targets: Targets) -> None:
        self.model = model
        self.trainer = trainer
        self.targets = targets

    def observed(self) -> dict[str, object]:
        """Each object's own account of itself, in one dict."""
        return {**self.model.applied(), **self.trainer.policies(), **self.targets.supervised()}

    def missing(self, asked: "Request") -> list[str]:
        """Every switch whose effect the objects do not show, named with both sides of the mismatch.

        ADAPTERS is the one asymmetric key: the run asks for a rank and the model holds a COUNT, so any
        positive count satisfies a positive request — the exact number of wrapped layers is the
        encoder's business, and `low_rank` already logs it. Zero against a positive rank is the failure
        that motivated this file.
        """
        observed = self.observed()
        wanted = asked.wanted()
        return [
            f"{key}: asked {want!r}, the model shows {observed[key]!r}"
            for key, want in wanted.items()
            if not self.agrees(key, observed[key], want)
        ]

    def agrees(self, key: str, observed: object, want: object) -> bool:
        """Whether one switch landed, compared the way that switch's own type can honestly be compared.

        A FLOAT IS NEVER COMPARED EXACTLY. `band_dense_llrd` died in its first second on
        `encoder_rate_decay: asked 0.7, the model shows 0.7000000000000004` — the decay is recovered from
        a per-layer rate the trainer computed by repeated multiplication, so the observed value is the
        asked one after a round trip through arithmetic, and exact equality reads that round trip as an
        unlanded switch. The gate is right to exist and its ruler was wrong: it cost a queue slot to a
        run whose switch had in fact landed, which is the same class of loss it was built to prevent.
        """
        if key in {ADAPTERS}:
            return bool(observed) == bool(want)
        if isinstance(want, float) and isinstance(observed, (int, float)) and not isinstance(observed, bool):
            return bool(np.isclose(float(observed), want, rtol=FLOAT_TOLERANCE, atol=FLOAT_TOLERANCE))
        return observed == want


@dataclass(frozen=True)
class Request:
    """What the run ASKED for, as one value rather than ten arguments.

    A value because these are not independent parameters: they are the switch settings of a single run,
    they are read together, and `wanted()` is the only thing anyone does with them.

    `compiling` is stored already resolved because the flag alone does not decide it — `main` compiles
    on CUDA only, so on CPU the honest request is False and a check reading the flag would fail every
    smoke run on this machine.

    Primitive fields rather than the options object: `TrainingOptions` builds both the model and the
    trainer, so a `Landed` that imported it would close an import cycle around the module it exists to
    protect.
    """

    lora_rank: int
    slice_regions: int
    recompute: bool
    compiling: bool
    bf16: bool
    whole_stack: bool
    slices: int
    train_size: int
    ema: bool
    one_cycle: bool
    reports: bool
    rank_weight: float
    distill_weight: float
    feature_weight: float
    dissent_weight: float
    contrast_weight: float
    auc_margin: float
    teaching: bool
    teaching_features: bool
    pair_memory: int
    per_label_loss: bool
    normalise_by_count: bool
    intensity_aug: bool
    weight_floor: bool
    calibrated_targets: bool
    finding_calibration: bool
    severity_weight: bool
    hold_annotations: bool
    grade_ladder: bool
    label_errors: str
    unread_targets: str
    side_weight: float
    focus_label: str
    unfocused_weight: float
    unfreeze_last: int
    accumulate: int
    layer_decay: float
    mixup: float
    weight_decay: float
    warmup: float
    grad_clip: float
    backbone_lr: float
    unfreeze_offset: int = NO_OFFSET
    """How many blocks at the TOP the request left frozen — zero for every arm before 2026-09-05."""

    sample_spread: Path | None = None
    sample_alpha: float = 0.0
    certainty_ramp: int = 0
    sample_ramp: int = 0
    corpus_oversample: float = 0.0
    target_corpus: str = ""
    slot_dropout: float = 0.0

    def wanted(self) -> dict[str, object]:
        """The request in the vocabulary the effects are reported in.

        THE TRANSLATION LIVES HERE AND NOWHERE ELSE: `--whole-stack` means a window of `slices` and its
        absence means `GROUP`, and a caller that re-derived that would be a second copy of the rule,
        free to disagree with the trainer's and to report a mismatch that is only its own.

        THE DEPTH IS MAPPED THROUGH THE ADAPTATION MODE, not read off the flag: under LoRA every block
        carries an adapter and `parameter_groups` ignores `unfreeze_last` deliberately, so the honest
        request there is zero open base blocks. Asking through the mode is what keeps this a strict
        equality — the alternative was a second loose key, which is the thing `ADAPTERS` had to be and
        which buys the check nothing anywhere else.

        THE DEPTH REGIONS ARE ASKED FOR AS THE FLAG and answered by the model with the count it will
        actually split on, which is 1 whenever another aggregator owns the slice axis. That asymmetry is
        the whole point: `--slice-regions 4 --slice-pooling attend` parsed, recorded and trained for the
        life of the flag while splitting nothing, so the ledger carried a fact about an arm that did not
        run. Here it is a plain equality that fails before the card is spent.

        THE ONE DELIBERATE EXCEPTION is the weight floor, whose bool-to-value rule `TrainingOptions`
        also spells. Taking the resolved `supervision.floor` instead would compare that property's
        output against itself and pass on any value it produced; re-deriving from the FLAG is what
        makes a broken `supervision` property visible here rather than only in the loss.
        """
        return {
            ADAPTERS: self.lora_rank,
            DEPTH_REGIONS: self.slice_regions,
            RECOMPUTING: self.recompute,
            COMPILED_ENCODER: self.compiling,
            AUTOCAST: torch.bfloat16 if self.bf16 else torch.float16,
            WINDOW: self.slices if self.whole_stack else GROUP,
            READ_SIZE: self.train_size or None,
            AVERAGING: self.ema,
            SCHEDULED: self.one_cycle,
            REPORTS: self.reports,
            RANK_TERM: self.rank_weight,
            DISTILL_TERM: self.distill_weight,
            FEATURE_TERM: self.feature_weight,
            DISSENT_TERM: self.dissent_weight,
            CONTRAST_TERM: self.contrast_weight,
            MARGIN_TERM: self.auc_margin,
            TEACHER: self.teaching,
            FEATURE_TEACHER: self.teaching_features,
            PAIR_POOL: self.pair_memory,
            PER_LABEL: self.per_label_loss,
            COUNT_NORMALISED: self.normalise_by_count,
            INTENSITY_AUGMENT: self.intensity_aug,
            WEIGHT_FLOOR: CONFIDENCE_FLOOR if self.weight_floor else 0.0,
            CALIBRATED: self.calibrated_targets,
            FINDING: self.finding_calibration,
            SEVERITY_WEIGHT: self.severity_weight,
            SUSPECT: self.label_errors,
            UNREAD: self.unread_targets,
            # Re-derived from the FLAG, and inverted here rather than in the options property, for the
            # reason WEIGHT_FLOOR gives above: taking `supervision.annotations` would compare that
            # property's output against itself and pass on whatever it produced.
            ANNOTATIONS: not self.hold_annotations,
            GRADE_LADDER: self.grade_ladder,
            SIDE_TERM: self.side_weight,
            FOCUS: self.focus_label,
            # RE-DERIVED FROM THE FLAG PAIR, as SAMPLING is: an unfocused weight with no findings named
            # scales nothing, so the honest request is the 0.0 the null policy accounts for.
            UNFOCUSED: self.unfocused_weight if self.focus_label else 0.0,
            # RE-DERIVED FROM THE FLAG PAIR, the way WINDOW is: an alpha without a spread file weights
            # nothing, so the honest request there is the uniform draw the run will actually get.
            SAMPLING: self.sample_alpha if self.sample_spread is not None else 0.0,
            SAMPLING_RAMP: self.sample_ramp if self.sample_spread is not None else 0,
            CORPUS_OVERSAMPLE: self.corpus_oversample,
            TARGET_CORPUS: self.target_corpus if self.corpus_oversample > 0.0 else "",
            CERTAINTY_RAMP: self.certainty_ramp,
            OPENED_BLOCKS: 0 if self.lora_rank else self.unfreeze_last,
            # WHERE the window sits, which the count above cannot see: an offset that reached
            # nothing opens the same NUMBER of blocks in the wrong place and reads as a null on
            # a mechanism that never ran. Zero whenever no base block is open at all.
            FROZEN_TOP: 0 if self.lora_rank or not self.unfreeze_last else self.unfreeze_offset,
            ACCUMULATION: self.accumulate,
            LAYER_DECAY: self.layer_decay,
            MIXUP: self.mixup,
            SLOT_DROPOUT: self.slot_dropout,
            # THE FOUR OPTIMISER-PATH POLICIES, and each is read back off the object that applies it
            # rather than off the value that was passed in: the decay from the optimiser's first
            # parameter group, the warmup from the field `one_cycle` hands to `pct_start`, the ceiling
            # from the `GradientClip` the step calls. A decay that never reached `AdamW` — the shape of
            # this failure, since it is a constructor keyword and a typo lands in no exception — reads
            # here as 0.02 against the value asked for.
            OPTIMISER_DECAY: self.weight_decay,
            WARMUP: self.warmup,
            CLIPPING: self.grad_clip,
            # THE ANCHOR, added 2026-09-05, and the reason LAYER_DECAY above was never enough on its
            # own: the decay is a RATIO, so it lands at exactly the asked 0.7 whether the rate it
            # divides is the one this run asked for or the one the recipe inherited.
            #
            # ASKED UNCONDITIONALLY, unlike OPENED_BLOCKS and FROZEN_TOP above, because the rate is a
            # property of the GROUP and not of what is in it: `parameter_groups` builds the backbone
            # group at this rate whether the depth flag opened four blocks, zero, or an adapter set,
            # so there is no configuration under which the honest request is anything but the flag.
            BACKBONE_RATE: self.backbone_lr,
        }
