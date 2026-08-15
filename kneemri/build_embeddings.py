"""Encode a pixel cache into an embedding cache: `python -m kneemri.build_embeddings`.

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
from transformers import AutoModel

from core.paths import DataRoot
from kneemri.backbone import ENCODERS, SMALL
from kneemri.embedding_cache import EmbeddingCache, EmbeddingSpec
from kneemri.model import CLS_MEAN, POOL_PARTS, StudyDesign, StudyModel
from kneemri.slots import CANONICAL_SCHEME, SCHEMES, Routing
from kneemri.study_cache import StudyCache

logger = logging.getLogger(__name__)


def main() -> None:
    """Build one embedding cache from one pixel cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default=SMALL, choices=tuple(ENCODERS))
    parser.add_argument("--image-size", type=int, default=336, help="the PIXEL cache to read")
    parser.add_argument("--train-size", type=int, default=None, help="resample to this before encoding")
    parser.add_argument("--slices", type=int, default=12)
    parser.add_argument("--scheme", default=CANONICAL_SCHEME, choices=tuple(SCHEMES))
    parser.add_argument(
        "--pool",
        default=CLS_MEAN,
        choices=tuple(POOL_PARTS),
        help="how patch tokens become one vector per slice. The name reaches the embedding cache KEY, "
        "so a pooling the model cannot build must fail here rather than after the backbone is fetched.",
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
    corpus, identifiers = StudyCache(
        root, slices=arguments.slices, image_size=arguments.image_size, routing=Routing(scheme=arguments.scheme)
    ).load()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Per-slice pooling, because the whole point is to keep each slice separate — the slot's own
    # aggregation is one of the choices this cache exists to make free.
    model = StudyModel(
        AutoModel.from_pretrained(encoder.repository),
        encoder.dim,
        len(corpus.mask[0]),
        StudyDesign(pool=arguments.pool, slice_pooling="max"),
    )
    model.eval().to(device)

    spec = EmbeddingSpec(
        variant=arguments.variant,
        image_size=arguments.image_size,
        read_size=read_size,
        slices=arguments.slices,
        scheme=arguments.scheme,
        pool=arguments.pool,
    )
    cache = EmbeddingCache(root, spec)
    logger.info(f"encoding {len(identifiers)} studies -> {cache.features_path}")
    with torch.no_grad():
        written = cache.build(model.slice_features, corpus, identifiers, device, batch=arguments.batch)
    logger.info(f"EMBEDDINGS BUILD COMPLETE {written.features.shape} width={written.width}")


if __name__ == "__main__":
    main()
