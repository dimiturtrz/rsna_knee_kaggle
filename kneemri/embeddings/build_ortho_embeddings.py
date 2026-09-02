"""Encode a pixel cache through OrthoDiffusion's three plane denoisers.

    python -m kneemri.embeddings.build_ortho_embeddings --image-size 336 --slices 12 --batch 4

The volumetric sibling of `build_embeddings`. That one walks the `ENCODERS` registry, whose entries are
2D HuggingFace trunks named by a repo and a width; this trunk is three local checkpoints selected by the
PLANE of each slot, which is not a variant of that idea and would have arrived there as a fourth mount
kind, a fake repo string and a branch inside `Encoder.load`.

It shares the part that matters — `EmbeddingCache`, whose `_probe_shape` already reads the encoder's own
output rank and so writes a depth-1 array for a whole-stack reading without being told. The result drops
straight into `feature_probe`, which is the only reason this exists: the question is whether a 3D
in-domain representation carries more of the twelve findings than our 2D trunks, and that is a probe, not
a member.
"""

import logging

import torch

from core import cli
from core.data.corpus.cache_flags import CacheFlags
from core.wakelock import Wakelock
from kneemri.device import VisibleCards
from kneemri.embeddings.embedding_cache import EmbeddingCache, EmbeddingSpec
from kneemri.models.ortho_diffusion import MIDDLE_WIDTH, READ_SIZE, WEIGHTS
from kneemri.models.ortho_weights import DenoiserWeights

logger = logging.getLogger(__name__)

EXTERNAL_KIND = "external"
"""The data root's directory for third-party weights — raptor's already lives there."""

VARIANT = "orthodiff"
"""The name this trunk's features are filed under, so `feature_probe.TRUNKS` can name the same stem."""

VOLUME = "volume"
"""The `slice_read` this build writes. A whole-stack reading consumes the slices inside the encoder, so
the array has a depth of 1 — a different rank from every per-slice cache, and the reason the field is in
the key at all."""

BATCH = 4
"""Studies per forward. Six slots each become a 16x256x256 volume through a 3D U-Net, so a batch of 8 is
six times the activation footprint of the 2D path at the same nominal size."""


def main() -> None:
    """Build one volumetric embedding cache from one pixel cache."""
    command = cli.Command(__doc__, logger)
    CacheFlags.add_to(command.parser)
    command.parser.add_argument("--weights", default=WEIGHTS, help="checkpoint directory under the external root")
    command.parser.add_argument("--batch", type=int, default=BATCH, help="studies per forward")
    arguments = command.parsed()

    rooted = command.rooted(arguments)
    root = rooted.processed()
    pixels = CacheFlags.cache(arguments, root)
    corpus, identifiers = pixels.load()

    device = VisibleCards().device()
    trunk = DenoiserWeights(rooted.kind(EXTERNAL_KIND) / arguments.weights).trunk(device)

    spec = EmbeddingSpec(
        variant=VARIANT,
        source=pixels.key,
        # Their channel multipliers are derived from 256, so the read size is a fact about the weights
        # rather than a flag. Recorded in the key anyway: a later trunk read at another size would
        # otherwise share this filename.
        read_size=READ_SIZE,
        pool=f"gap{MIDDLE_WIDTH}",
        slice_read=VOLUME,
    )
    cache = EmbeddingCache(root, spec)
    logger.info(f"encoding {len(identifiers)} studies -> {cache.features_path}")
    with torch.no_grad(), Wakelock():
        written = cache.build(trunk.slice_features, corpus, identifiers, device, batch=arguments.batch)
    logger.info(f"ORTHO EMBEDDINGS BUILD COMPLETE {written.features.shape} width={written.width}")


if __name__ == "__main__":
    main()
