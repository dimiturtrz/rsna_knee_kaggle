"""Adapt the encoder to knee MRI without labels: `python -m kneemri.training.pretrain --epochs N`.

The corpus is the decode-once cache the supervised trainer already reads, taken one SLICE at a time
rather than one study at a time. That is the whole difference in what the data means here: supervision
consumes a study as twelve slices of three slots carrying twelve labels between them, so the label is
the scarce thing and the pixels are spent on producing one gradient per study. An unlabelled objective
has one target per slice, so the same cache is roughly two orders of magnitude more training signal.

WHAT THIS DOES NOT DO, stated because the bead's original framing overclaimed it and was corrected: this
reads the SAME cached pixels, not the out-of-band slices the band excludes. Widening the band means a
99 GB rebuild, and it is a second variable. One at a time.

The output is a plain encoder state dict, keyed exactly as `AutoModel.from_pretrained` keys it, so the
supervised recipe consumes it through `--encoder-weights` with nothing else changed and
`kneemri.evaluation.encoder_drift` can measure it against the same stock reference.
"""

import argparse
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from torch.amp.autocast_mode import autocast

from core.data.corpus.cache_flags import CacheFlags
from core.data.corpus.study_cache import CachedCorpus
from core.paths import DataRoot
from core.wakelock import Wakelock
from kneemri.models.backbone import ENCODERS, SMALL
from kneemri.training.spectral_mae import BYTE_RANGE, SpectralMae, SpectralMaskConfig

logger = logging.getLogger(__name__)

PRESENT = 0.5
"""A slot's presence mask is a float coverage figure, not a flag; anything at least this covered is read.

The same threshold the slot head uses, so the corpus this trains on is the corpus supervision sees."""

REPORT_EVERY = 50


@dataclass(frozen=True)
class Optimisation:
    """The optimiser and its schedule, which are one thing.

    One thing because they are never correct apart: a schedule stepped without its optimiser silently
    advances the rate against no update, and an optimiser stepped without its schedule runs the whole
    pretraining at the peak rate — the second of which produces a plausible loss curve and a ruined
    encoder. They were being threaded through two methods as a pair, which is the clump saying so.
    """

    optimiser: torch.optim.Optimizer
    schedule: torch.optim.lr_scheduler.LRScheduler

    @classmethod
    def one_cycle(cls, parameters: Iterable[Tensor], lr: float, steps: int) -> "Optimisation":
        """The schedule the supervised recipe settled on, so pretraining is not a second unknown."""
        optimiser = torch.optim.AdamW(parameters, lr=lr)
        return cls(optimiser, torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=lr, total_steps=steps))

    def apply(self, loss: Float[Tensor, ""]) -> None:
        """Backward, step, advance — in the one order that is correct."""
        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        self.optimiser.step()
        self.schedule.step()


@dataclass(frozen=True)
class Sampling:
    """How slices are drawn and how far an epoch runs.

    Three fields that are one decision — the batch width, the stream they are drawn from, and where the
    epoch stops — and they were arriving as three positional arguments beside the model, the corpus and
    the device, which is two unrelated subjects in one signature.
    """

    slices_per_step: int
    generator: torch.Generator
    max_steps: int | None = None


class Pretrainer:
    """One masked-reconstruction run over the cached slices.

    Holds the objective, the optimiser and the sampler together because all three are decided by the same
    two numbers — how many slices a step sees and how many steps there are — and splitting them left the
    step count being recomputed in three places.
    """

    def __init__(
        self,
        objective: SpectralMae,
        corpus: CachedCorpus,
        device: torch.device,
        sampling: Sampling,
    ) -> None:
        self.objective = objective
        self.corpus = corpus
        self.device = device
        self.sampling = sampling
        # A FLAT LIST OF PRESENT (study, slot) PAIRS, built once. Sampling studies and then discarding
        # their absent slots would make the batch size a random variable, and the absent slots are not
        # uniformly spread — a scheme's third slot is missing far more often than its first — so the
        # discard would silently reweight the corpus toward the studies that acquired everything.
        studies, slots = np.nonzero(corpus.mask >= PRESENT)
        self.pairs = np.stack([studies, slots], axis=1)
        self.per_pair = int(corpus.images.shape[2])

    def steps_per_epoch(self) -> int:
        """How many steps cover the corpus once, at the configured slices per step.

        `max_steps` truncates that, and exists for one reason: a CLI whose wiring is only exercised by
        the forty-minute run it was meant to prove is a CLI discovered broken by the queue slot. It is a
        SMOKE knob — a run that sets it has not seen the corpus once and its weights mean nothing.
        """
        full = max(1, len(self.pairs) * self.per_pair // self.sampling.slices_per_step)
        capped = self.sampling.max_steps
        return min(full, capped) if capped else full

    def batch(self) -> Float[Tensor, "b 1 h w"]:
        """One batch of independent slices, as float in [0, 1] with a single channel.

        Slices are drawn INDEPENDENTLY rather than as whole slots. Adjacent slices of one slot are nearly
        the same image, so a batch built from whole slots would carry a fraction of the independent
        samples its size claims, and the reconstruction task would be solvable by copying a neighbour
        that happens to sit in the same batch.
        """
        width, stream = self.sampling.slices_per_step, self.sampling.generator
        rows = torch.randint(len(self.pairs), (width,), generator=stream).numpy()
        depth = torch.randint(self.per_pair, (width,), generator=stream).numpy()
        chosen = self.pairs[rows]
        pixels = self.corpus.images[chosen[:, 0], chosen[:, 1], depth]
        images = torch.from_numpy(np.ascontiguousarray(pixels)).to(self.device, non_blocking=True)
        return images.unsqueeze(1).float().div_(BYTE_RANGE)

    def step(self, optimisation: "Optimisation") -> tuple[float, float]:
        """One optimiser step, returning the pixel and spectral terms separately.

        Separately because the spectral term IS the claim being tested: a total that falls while the
        spectral term is flat means the run is a plain masked autoencoder wearing the name.
        """
        images = self.batch()
        tokens = (images.shape[-1] // self.objective.config.patch) ** 2
        mask = self.objective.sample_mask(images.shape[0], tokens, self.device, self.sampling.generator)
        with autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            total, pixels, spectral = self.objective(images, mask)
        optimisation.apply(total)
        return float(pixels.detach()), float(spectral.detach())

    def run(self, epochs: int, optimisation: "Optimisation") -> None:
        """Every epoch, logging both loss terms and the throughput each epoch actually achieved."""
        steps = self.steps_per_epoch()
        for epoch in range(epochs):
            started = time.monotonic()
            totals = torch.zeros(2)
            for index in range(steps):
                pixels, spectral = self.step(optimisation)
                totals += torch.tensor([pixels, spectral])
                if index % REPORT_EVERY == 0:
                    logger.info("  step %d/%d  pixel %.4f  spectral %.4f", index, steps, pixels, spectral)
            elapsed = time.monotonic() - started
            logger.info(
                "epoch %d: pixel %.4f  spectral %.4f  (%.0fs, %.0f slices/s)",
                epoch,
                totals[0] / steps,
                totals[1] / steps,
                elapsed,
                steps * self.sampling.slices_per_step / elapsed,
            )

    def save(self, destination: Path, variant: str, epochs: int, cache: str) -> None:
        """The adapted encoder plus the recipe that produced it, under one path.

        The recipe travels WITH the weights because a bare state dict cannot answer the question the
        write-up will ask first — whether the flag being credited actually landed. That has now cost this
        project enough arms to be a rule rather than a preference.

        THE CACHE KEY IS PART OF THE RECIPE, and its absence was the gap this docstring did not cover.
        The objective's own knobs were recorded and the SLICE POPULATION was not, so an adaptation could
        not say which pixels it had seen. On 2026-08-20 both adaptations on disk turned out to have been
        run on the DEFAULT cache — 12 slices, band 0.20-0.80 — while every arm importing them trains on
        20 slices at 0.10-0.90, and the only place that fact survived was a line in a launch log. The key
        names everything that decides the cache's contents, so recording it makes the mismatch a
        checkable value rather than an archaeology problem.
        """
        torch.save(
            {
                "encoder": self.objective.encoder_state(),
                "variant": variant,
                # Beside `variant`, not inside `config`: both are facts about the adaptation as a whole,
                # where `config` holds the objective's knobs. Keeping it out of `config` also keeps it
                # out of the vocabulary a MEMBER checkpoint's config is scraped against — a different
                # file's key that happened to share the name would read there as an unstamped one.
                "cache": cache,
                "config": vars(self.objective.config)
                | {"epochs": epochs, "slices_per_step": self.sampling.slices_per_step},
            },
            destination,
        )
        logger.info("wrote %s", destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    CacheFlags.add_to(parser)
    parser.add_argument("--variant", default=SMALL, help="which encoder to adapt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--slices-per-step",
        type=int,
        default=96,
        help="slices in one optimiser step. The supervised recipe moves 432 slices per step through the "
        "encoder AND a slot head at 26.9 GB; this carries no head, so the ceiling is higher and the "
        "reason to stay under it is that a masked-reconstruction gradient wants many steps, not few.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0e-4,
        help="peak encoder rate. Above the 4e-5 the supervised recipe settled on, because that value was "
        "chosen to keep a 12-epoch fine-tune from destroying a representation it cannot rebuild — the "
        "opposite of what this run is for.",
    )
    parser.add_argument("--mask-ratio", type=float, default=SpectralMaskConfig.mask_ratio)
    parser.add_argument("--spectral-weight", type=float, default=SpectralMaskConfig.spectral_weight)
    parser.add_argument("--highpass", type=float, default=SpectralMaskConfig.highpass)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="SMOKE ONLY: truncate each epoch to this many steps, to prove the wiring without spending "
        "the card. A checkpoint written under this flag has not seen the corpus once.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from transformers import AutoModel  # noqa: PLC0415 - a 90 MB import the unit gate must not pay for

    encoder = ENCODERS[arguments.variant]
    root = DataRoot.from_config(arguments.paths)
    cache = CacheFlags.cache(arguments, root.processed())
    corpus, identifiers = cache.load()
    encoder.require_tiling(corpus.image_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = SpectralMaskConfig(
        mask_ratio=arguments.mask_ratio,
        spectral_weight=arguments.spectral_weight,
        highpass=arguments.highpass,
        patch=encoder.patch,
    )
    objective = SpectralMae(AutoModel.from_pretrained(encoder.repository), encoder.dim, config).to(device)
    if device.type == "cuda":
        objective.backbone.compile()

    sampling = Sampling(
        slices_per_step=arguments.slices_per_step,
        generator=torch.Generator().manual_seed(arguments.seed),
        max_steps=arguments.max_steps,
    )
    trainer = Pretrainer(objective, corpus, device, sampling)
    steps = trainer.steps_per_epoch() * arguments.epochs
    optimisation = Optimisation.one_cycle(objective.parameters(), arguments.lr, steps)

    logger.info(
        "%d studies, %d present slots, %d slices per slot -> %d slices; %d steps of %d",
        len(identifiers),
        len(trainer.pairs),
        trainer.per_pair,
        len(trainer.pairs) * trainer.per_pair,
        steps,
        arguments.slices_per_step,
    )
    with Wakelock():
        trainer.run(arguments.epochs, optimisation)
    trainer.save(arguments.out, arguments.variant, arguments.epochs, cache.key)


if __name__ == "__main__":
    main()
