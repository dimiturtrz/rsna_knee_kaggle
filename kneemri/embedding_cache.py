"""Encode the corpus once, so every question that is not about pixels costs seconds instead of hours.

A training step spends essentially all of its time in the encoder: 4,407 studies x 6 slots x 12 slices
is 317k images per epoch, and at the measured 940 img/s that is 320 s of card time to ask ANY question
— including the ones that never touch a pixel. Slot-head design, the anatomical prior, the rank weight,
the weight floor, the loss shape, how slices are pooled: all of them are cheap functions of the
encoder's output, and all of them currently cost a full arm to try.

The encoder's output for a FROZEN encoder does not change between those runs. Computing it once and
storing it turns a 66-minute arm into a few matrix multiplies. The whole fold is 317k x 768 float16 =
487 MB — small enough to hold in RAM, let alone on disk.

WHAT THIS IS NOT. It is a PROXY. A frozen encoder is a different function from one whose top six blocks
adapt, so a head tuned on these features is not guaranteed to be the head that wins after fine-tuning.
The harness is worth trusting only once that correlation is measured — take head variants whose
fine-tuned scores are already known, rank them here, and check the ORDERING survives. If it does not,
this answers a different question than the one being asked, and should be dropped rather than believed.

SLICES ARE KEPT SEPARATE, deliberately. Pooling them here would bake in the one head-side choice most
worth sweeping; storing per slice keeps max, attention and anything later as free choices over the same
tensor, at the cost of a factor of `slices` in size — which the arithmetic above says we can afford.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, UInt8
from numpy.lib.format import open_memmap
from numpy.typing import NDArray
from torch import Tensor

from kneemri.study_cache import CachedCorpus

logger = logging.getLogger(__name__)

SliceFeatures = Callable[[UInt8[Tensor, "b s g p q"], int | None], Float[Tensor, "b s g w"]]
"""What this module needs of a model: ONE call, not a model.

`StudyModel.slice_features` bound is exactly this, and so is any stand-in — which keeps the encode path
from importing the model to name a type, and keeps a compiled or wrapped model working unchanged. It is
an alias rather than a Protocol because a Protocol is a class, and this file already has its subject.
"""

STORED = np.float16
"""Half precision, because these are read straight into an autocast forward that would cast them down
anyway, and because it halves a file that is otherwise a gigabyte per configuration."""

PROGRESS_EVERY = 20
"""Batches between heartbeats. A silent 6-minute encode is indistinguishable from a hung one."""


@dataclass(frozen=True)
class EmbeddedCorpus:
    """The encoder's output for every slice of every slot, with the presence mask that indexes it.

    The sibling of `CachedCorpus` one stage further along: that one holds pixels, this one holds what
    the encoder made of them. The mask is carried for the same reason — a feature vector for a slot
    that was never acquired is zeros, and only the mask can say so.
    """

    features: NDArray[np.float16]
    mask: NDArray[np.float32]

    @property
    def width(self) -> int:
        """The pooled feature width, read off the array rather than recomputed from a registry."""
        return int(self.features.shape[-1])

    @property
    def slices(self) -> int:
        """How many slices per slot were encoded, which bounds what a pooling can read."""
        return int(self.features.shape[2])


@dataclass(frozen=True)
class EmbeddingSpec:
    """Everything that decides what the features ARE, as one value rather than seven arguments.

    Seven positional facts about a cache is exactly the shape that gets called with two of them
    transposed, and a transposed `image_size`/`read_size` produces a file that loads fine and describes
    different pixels. Grouped so the key, the paths and the build all read the same object.
    """

    variant: str
    image_size: int
    read_size: int
    slices: int
    scheme: str
    pool: str


class EmbeddingCache:
    """One encoded corpus on disk: built once from a pixel cache, read by every head experiment."""

    def __init__(self, directory: Path, spec: EmbeddingSpec) -> None:
        self.directory = directory
        self.spec = spec

    @property
    def key(self) -> str:
        """Everything that decides these features, in the filename.

        The pixel cache learned this lesson expensively: its name once omitted the slot count, so an
        extended-scheme build would have written a differently-shaped array over the six-slot cache
        every member was fitted on. The same failure is available here and is worse, because features
        carry no visible sign of which encoder produced them — a DINOv3 file and a DINOv2 file of the
        same shape load into the same head and score differently for no discoverable reason.

        The READ size is separate from the cache's size: our best recipe holds a 448 px cache and reads
        it at 392, and features computed at one are not features computed at the other.
        """
        spec = self.spec
        parts = [spec.variant, f"{spec.image_size}px", f"r{spec.read_size}", f"{spec.slices}sl", spec.pool]
        if spec.scheme != "canonical":
            parts.append(spec.scheme)
        return "_".join(parts)

    @property
    def features_path(self) -> Path:
        """Where the encoded features live."""
        return self.directory / f"features_{self.key}.npy"

    @property
    def mask_path(self) -> Path:
        """The presence mask, copied beside the features so the pair cannot be separated."""
        return self.directory / f"features_mask_{self.key}.npy"

    @property
    def index_path(self) -> Path:
        """The study order, which makes a feature row traceable back to a study."""
        return self.directory / f"features_index_{self.key}.json"

    def exists(self) -> bool:
        """Whether all three artefacts are present.

        All three, not just the features: a features file without its index is a matrix nobody can
        align to labels, and it would otherwise read as a usable cache.
        """
        return all(path.exists() for path in (self.features_path, self.mask_path, self.index_path))

    def build(
        self,
        encode: SliceFeatures,
        corpus: CachedCorpus,
        identifiers: list[str],
        device: torch.device,
        batch: int = 8,
    ) -> EmbeddedCorpus:
        """Encode every study once and write the result, returning what was written.

        NO GRADIENTS AND NO AUGMENTATION, both on purpose. The features are a fixed function of the
        pixels — that is the entire premise — so an augmented encode would produce one arbitrary draw
        of an augmentation and freeze it, which is worse than no augmentation because it looks like
        data.

        Written through a memmap rather than accumulated in memory: a nine-slot 448 px configuration is
        several GB, and the failure mode of getting that wrong is the OS killing the process with no
        error, which this repository has already paid for once.
        """
        studies, slots = corpus.mask.shape
        width = self._probe_width(encode, corpus, device)
        features = open_memmap(
            self.features_path, mode="w+", dtype=STORED, shape=(studies, slots, self.spec.slices, width)
        )
        with torch.no_grad(), torch.autocast("cuda", enabled=device.type == "cuda"):
            for start in range(0, studies, batch):
                stop = min(start + batch, studies)
                block = np.ascontiguousarray(corpus.images[start:stop, :, : self.spec.slices])
                pixels = torch.from_numpy(block).to(device)
                encoded = encode(pixels, self.spec.read_size)
                features[start:stop] = encoded.float().cpu().numpy().astype(STORED)
                if (start // batch) % PROGRESS_EVERY == 0:
                    logger.info("encoded %d/%d studies", stop, studies)
        features.flush()
        np.save(self.mask_path, corpus.mask)
        self.index_path.write_text(json.dumps(identifiers), encoding="utf-8")
        logger.info("wrote %s: %s", self.features_path.name, features.shape)
        return EmbeddedCorpus(features, corpus.mask)

    def _probe_width(self, encode: SliceFeatures, corpus: CachedCorpus, device: torch.device) -> int:
        """The feature width, taken from the model rather than from a registry.

        Derived by encoding ONE study rather than by multiplying a registry's `dim` by a pooling's part
        count — those are two places that can disagree, and this file's whole value depends on the array
        on disk matching the model that will later read it.
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=device.type == "cuda"):
            probe = torch.from_numpy(np.ascontiguousarray(corpus.images[:1, :, : self.spec.slices])).to(device)
            return int(encode(probe, self.spec.read_size).shape[-1])

    def load(self) -> tuple[EmbeddedCorpus, list[str]]:
        """The encoded corpus, memory-mapped read-only, with its study order."""
        if not self.exists():
            message = f"no embedding cache at {self.features_path} — build it first"
            raise FileNotFoundError(message)
        features = np.load(self.features_path, mmap_mode="r")
        mask = np.load(self.mask_path)
        identifiers = json.loads(self.index_path.read_text(encoding="utf-8"))
        return EmbeddedCorpus(features, mask), identifiers
