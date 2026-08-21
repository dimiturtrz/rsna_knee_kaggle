"""Encode a pixel cache into an embedding cache: `python -m kneemri.embeddings.build_embeddings`.

The counterpart to `build_cache`, one stage further along. That one turns DICOM into pixels; this one
turns pixels into the encoder's output, once, so every head-side question afterwards is arithmetic
rather than an hour of card time (`embedding_cache`).

THE ENCODER IS FROZEN AND UNTRAINED HERE ON PURPOSE. These features are the pretrained encoder's, not
a fitted member's: the harness exists to rank head designs cheaply BEFORE a member is fitted, and
features from a member already fitted on this fold would be reporting its training data back to it.
"""

import argparse
import logging
from pathlib import Path

import torch

from core.paths import DataRoot
from core.wakelock import Wakelock
from kneemri.corpus.cache_flags import CacheFlags
from kneemri.device import VisibleCards
from kneemri.embeddings.embedding_cache import EmbeddingCache, EmbeddingSpec
from kneemri.models.backbone import ENCODERS, SMALL
from kneemri.models.model import StudyDesign, StudyModel
from kneemri.models.slice_pooling import MAXIMUM, SLICE_POOLINGS
from kneemri.models.token_pooling import CLS_MEAN, POOL_PARTS

logger = logging.getLogger(__name__)


def main() -> None:
    """Build one embedding cache from one pixel cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default=SMALL, choices=tuple(ENCODERS))
    CacheFlags.add_to(parser)
    parser.add_argument("--train-size", type=int, default=None, help="resample to this before encoding")
    parser.add_argument(
        "--pool",
        default=CLS_MEAN,
        choices=tuple(POOL_PARTS),
        help="how patch tokens become one vector per slice. The name reaches the embedding cache KEY, "
        "so a pooling the model cannot build must fail here rather than after the backbone is fetched.",
    )
    parser.add_argument(
        "--slice-read",
        default=MAXIMUM,
        choices=tuple(SLICE_POOLINGS),
        help="how the slices of a slot reach the encoder. A per-slice reading keeps each slice separate "
        "and stores a slice axis for a head trial to reduce; a whole-stack reading consumes the slices "
        "inside the encoder and stores one vector per slot. It reaches the cache KEY, because the two "
        "produce arrays of different rank from the same pixels.",
    )
    parser.add_argument("--batch", type=int, default=8, help="studies per forward")
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    encoder = ENCODERS[arguments.variant]
    read_size = arguments.train_size or arguments.image_size
    # Before anything is allocated or decoded, for the same reason the trainer checks it there: a size
    # the patch grid cannot tile is either a crash deep in the encoder or a silent interpolation onto a
    # grid the weights never saw.
    encoder.require_tiling(read_size)

    root = DataRoot.from_config(arguments.paths).processed()
    # THE WHOLE READING, not the three axes this CLI happened to take flags for. It could not be pointed
    # at the 80 mm arrays at all, so it encoded the default ones whatever was asked of it.
    pixels = CacheFlags.cache(arguments, root)
    corpus, identifiers = pixels.load()

    device = VisibleCards().device()
    # Per-slice by DEFAULT, because keeping each slice separate makes the slot's own aggregation a free
    # choice later. It is a flag rather than a constant because a volumetric trunk REFUSES a per-slice
    # reading — it would spend the trunk's only advantage — so hard-coding it here put the one axis a
    # volume-pretrained encoder needs beyond this CLI's reach.
    model = StudyModel(
        encoder.load(),
        encoder.dim,
        len(corpus.mask[0]),
        StudyDesign(pool=arguments.pool, slice_pooling=arguments.slice_read),
    )
    model.eval().to(device)

    spec = EmbeddingSpec(
        variant=arguments.variant,
        source=pixels.key,
        read_size=read_size,
        pool=arguments.pool,
        slice_read=arguments.slice_read,
    )
    cache = EmbeddingCache(root, spec)
    logger.info(f"encoding {len(identifiers)} studies -> {cache.features_path}")
    # The encode pass is the same unattended hour-scale job the decode is, and suspends the same way.
    with torch.no_grad(), Wakelock():
        written = cache.build(model.slice_features, corpus, identifiers, device, batch=arguments.batch)
    logger.info(f"EMBEDDINGS BUILD COMPLETE {written.features.shape} width={written.width}")


if __name__ == "__main__":
    main()
