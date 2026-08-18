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

import torch

from kneemri.corpus.slot_reader import GROUP
from kneemri.models.model import ADAPTERS, COMPILED_ENCODER, RECOMPUTING, StudyModel
from kneemri.training.trainer import AUTOCAST, AVERAGING, READ_SIZE, REPORTS, SCHEDULED, WINDOW, Trainer


class Landed:
    """What a built model and trainer BECAME, against what the run asked them to be."""

    def __init__(self, model: StudyModel, trainer: Trainer) -> None:
        self.model = model
        self.trainer = trainer

    def observed(self) -> dict[str, object]:
        """Both objects' own accounts of themselves, in one dict."""
        return {**self.model.applied(), **self.trainer.policies()}

    def missing(self, asked: "Request") -> list[str]:
        """Every switch whose effect the objects do not show, named with both sides of the mismatch.

        ADAPTERS is the one asymmetric key: the run asks for a rank and the model holds a COUNT, so any
        positive count satisfies a positive request — the exact number of wrapped layers is the
        encoder's business, and `low_rank` already logs it. Zero against a positive rank is the failure
        that motivated this file.
        """
        observed = self.observed()
        wanted = asked.wanted()
        loose = {ADAPTERS}
        return [
            f"{key}: asked {want!r}, the model shows {observed[key]!r}"
            for key, want in wanted.items()
            if not (bool(observed[key]) == bool(want) if key in loose else observed[key] == want)
        ]


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
    recompute: bool
    compiling: bool
    bf16: bool
    whole_stack: bool
    slices: int
    train_size: int
    ema: bool
    one_cycle: bool
    reports: bool

    def wanted(self) -> dict[str, object]:
        """The request in the vocabulary the effects are reported in.

        THE TRANSLATION LIVES HERE AND NOWHERE ELSE: `--whole-stack` means a window of `slices` and its
        absence means `GROUP`, and a caller that re-derived that would be a second copy of the rule,
        free to disagree with the trainer's and to report a mismatch that is only its own.
        """
        return {
            ADAPTERS: self.lora_rank,
            RECOMPUTING: self.recompute,
            COMPILED_ENCODER: self.compiling,
            AUTOCAST: torch.bfloat16 if self.bf16 else torch.float16,
            WINDOW: self.slices if self.whole_stack else GROUP,
            READ_SIZE: self.train_size or None,
            AVERAGING: self.ema,
            SCHEDULED: self.one_cycle,
            REPORTS: self.reports,
        }
