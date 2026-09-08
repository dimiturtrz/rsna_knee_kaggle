"""Where a SAVED adaptation's attention actually sits — on the class token, on the registers, or on patches.

`pretrain` prints this reading once an epoch and then the run ends, so a checkpoint on disk cannot be
asked the question again. That was survivable while the only statistic was `prefix_mass`, which the
epoch lines already carried. It stopped being survivable the moment `class_mass` existed: a live process
froze its code at launch, so an arm running when the split was added prints the summed number for its
whole life, and the split it was queued to measure is readable ONLY off the weights it leaves behind.

That is this module's entire reason to exist, and it is the general form of a defect this project has
paid for repeatedly — a reading taken inside a run, by a process, that nothing can reproduce afterwards.
A checkpoint plus a probe batch is a complete description of an encoder's attention, so the reading
belongs to the WEIGHTS and not to the process that happened to be alive when they were written.
"""

import logging
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import torch
from jaxtyping import Float
from torch import Tensor

from core import cli
from core.data.corpus.cache_flags import CacheFlags
from kneemri.models.backbone import ENCODERS, Encoder
from kneemri.training.attention_health import AttentionHealth, AttentionProfile
from kneemri.training.masked_objective import MaskConfig, MaskedObjective
from kneemri.training.masked_objectives import Objectives
from kneemri.training.pretrain import Pretrainer, Sampling

logger = logging.getLogger(__name__)

PROBE_SLICES = 96
"""How many slices the reading is averaged over — the pretrainer's own default step width.

The same number as the run used, so a profile read here and an epoch line read there are averages over
batches of one size. It is a mean over independent slices either way, but a mean over 96 and a mean over
8 are not equally steady, and the whole point of this module is that its numbers sit beside those.
"""


class Adaptation(TypedDict):
    """The four keys `pretrain.save` writes, named so the reader is not a dictionary of anything.

    A checkpoint is a JSON-shaped boundary — it crosses a process and a week — and that is exactly where
    a TypedDict earns its keep over a bare mapping: the misuse it prevents is reading `state["cache"]`
    on a file that never carried one, which is what every adaptation written before `save` recorded the
    key actually is.
    """

    variant: str
    cache: str
    config: dict[str, float]
    encoder: dict[str, Tensor]


@dataclass(frozen=True)
class SavedAdaptation:
    """One `pretrain` checkpoint read back as what it is — weights plus the recipe that produced them.

    Everything except the objective's NAME was written down beside the weights, so nothing here needs a
    caller to remember it. That matters more than convenience: a flag a caller retypes is a flag a
    caller can retype WRONG, and this project has already spent two adaptations on exactly that (both
    were adapted on the default 12-slice cache while every arm importing them trained on 20 slices).
    """

    state: Adaptation

    @classmethod
    def load(cls, checkpoint: Path) -> "SavedAdaptation":
        """The file, with its recipe — `weights_only=False` because the recipe is what makes it usable."""
        return cls(cast("Adaptation", torch.load(checkpoint, map_location="cpu", weights_only=False)))

    @property
    def encoder(self) -> Encoder:
        """Which trunk these weights are, named by the checkpoint rather than by a flag."""
        return ENCODERS[self.state["variant"]]

    @property
    def recipe(self) -> Namespace:
        """The recorded `vars(config)`, shaped as the namespace an objective builder reads its flags from."""
        return Namespace(**self.state["config"])

    def require_cache(self, key: str) -> None:
        """Refuse a probe drawn from a different slice population than the one the weights were fitted on.

        A profile on other pixels is a reading of another experiment. The key is checkable only because
        `save` wrote it down, which is the whole argument for writing recipes beside weights.
        """
        if key != self.state["cache"]:
            message = (
                f"adapted on cache {self.state['cache']!r}, and these flags resolve to {key!r}; "
                f"a profile on different pixels is not a reading of this checkpoint"
            )
            raise RuntimeError(message)

    def objective(  # devtools-ignore: test-mirror - building one downloads a published checkpoint
        self, name: str, device: torch.device
    ) -> MaskedObjective[MaskConfig]:
        """The objective rebuilt on its own recipe, carrying these weights instead of the published ones."""
        built = Objectives.named(name, self.recipe, self.encoder).to(device)
        built.backbone.load_state_dict(self.state["encoder"])
        return built

    def profile(  # devtools-ignore: test-mirror - needs a real trunk, which hygiene.small forbids
        self, objective: MaskedObjective[MaskConfig], images: Float[Tensor, "b 1 h w"]
    ) -> AttentionProfile:
        """The reading, against no reference: both masses are absolute shares of a softmax row.

        Only `retention` needs a stock encoder to be read against, and this is not that gate — the gate
        already ran, inside the run being profiled. Downloading a second trunk to divide by would add a
        number nobody asked for and a network dependency to a read of a local file.
        """
        tokens = (images.shape[-1] // objective.config.patch) ** 2
        health = AttentionHealth(tokens=tokens, reference=AttentionProfile((), ()), probe=objective.normalise(images))
        return health.profile(objective.backbone)


def main() -> None:
    command = cli.Command(__doc__, logger)
    CacheFlags.add_to(command.parser)
    command.parser.add_argument("--checkpoint", type=Path, required=True, help="an adaptation written by pretrain")
    command.parser.add_argument(
        "--objective",
        choices=Objectives.choices(),
        default="latent",
        help="the one thing the checkpoint does not record. It decides the input units and the token "
        "count that the weights are read through, never the weights themselves.",
    )
    # `--slices-per-step`, spelled exactly as `pretrain` spells it, and NOT `--slices`: that name is
    # already CacheFlags', where it means how many slices a study contributes to the CACHE. The two are
    # different populations and argparse refuses the collision, which is the correct outcome — the flag
    # that picks the probe's width must not be confusable with the one that decides which pixels exist.
    command.parser.add_argument("--slices-per-step", type=int, default=PROBE_SLICES, help="probe batch width")
    command.parser.add_argument("--seed", type=int, default=2026, help="which slices the probe batch draws")
    command.parser.add_argument(
        "--device",
        default="cpu",
        help="cpu by default, because this is one forward pass and the card is usually holding an arm. "
        "A profile is not a throughput measurement and has no reason to contend for it.",
    )
    arguments = command.parsed()

    saved = SavedAdaptation.load(arguments.checkpoint)
    cache = CacheFlags.cache(arguments, command.rooted(arguments).processed())
    saved.require_cache(cache.key)
    corpus, _ = cache.load()
    saved.encoder.require_tiling(corpus.image_size)

    device = torch.device(arguments.device)
    objective = saved.objective(arguments.objective, device)
    sampling = Sampling(
        slices_per_step=arguments.slices_per_step,
        generator=torch.Generator().manual_seed(arguments.seed),
        max_steps=None,
    )
    images = Pretrainer(objective, corpus, device, sampling).batch()

    logger.info("%s on %d slices of %s", arguments.checkpoint.name, arguments.slices_per_step, saved.state["cache"])
    logger.info("%s", saved.profile(objective, images).report())


if __name__ == "__main__":
    main()
