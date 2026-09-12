"""Encode a pixel cache through MedicalNet's 3D ResNet-50.

    python -m kneemri.embeddings.build_medicalnet_embeddings --image-size 336 --slices 12 --batch 4

The second volumetric build, and it exists to disambiguate the first. `build_ortho_embeddings` wrote the
only volumetric cache this project has ever probed, and that probe LOST (`orthodiff@mean` 0.7435 solo,
+0.0001 into `honest12`) on a 256-wide readout its own module pre-registered as the confounded direction.
This one reads 2048 wide off the same six slots at the same depth and size, so the trunk is the only term
that differs. The bands are pre-registered in `kneemri.models.medicalnet`, not here.

It shares the part that matters with its sibling — `EmbeddingCache`, whose `_probe_shape` reads the
encoder's own output rank and so writes a depth-1 array for a whole-stack reading without being told.
"""

import logging

import torch

from core import cli
from core.data.corpus.cache_flags import CacheFlags
from core.wakelock import Wakelock
from kneemri.device import VisibleCards
from kneemri.embeddings.embedding_cache import EmbeddingCache, EmbeddingSpec
from kneemri.models.medicalnet import READ_SIZE, WEIGHTS, WIDTH
from kneemri.models.medicalnet_weights import MedicalNetWeights

logger = logging.getLogger(__name__)

EXTERNAL_KIND = "external"
"""The data root's directory for third-party weights — raptor's and OrthoDiffusion's already live there."""

VARIANT = "medicalnet"
"""The name this trunk's features are filed under, so `feature_probe.TRUNKS` can name the same stem."""

VOLUME = "volume"
"""The `slice_read` this build writes. A whole-stack reading consumes the slices inside the encoder, so
the array has a depth of 1 — a different rank from every per-slice cache, and the reason the field is in
the key at all."""

BATCH = 4
"""Studies per forward. Matched to the ortho build: six slots each become a 16x256x256 volume, and
`layer3`/`layer4` run DILATED at stride 1, so the deepest activations stay at 2x32x32 rather than
collapsing the way a classification ResNet's would."""


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
    trunk = MedicalNetWeights(rooted.kind(EXTERNAL_KIND) / arguments.weights).trunk(device)

    spec = EmbeddingSpec(
        variant=VARIANT,
        source=pixels.key,
        # Pinned to the sibling probe rather than flagged, so the two volumetric caches differ in the
        # trunk alone. Recorded in the key anyway: a later read at another size would share this name.
        read_size=READ_SIZE,
        pool=f"gap{WIDTH}",
        slice_read=VOLUME,
    )
    cache = EmbeddingCache(root, spec)
    logger.info(f"encoding {len(identifiers)} studies -> {cache.features_path}")
    with torch.no_grad(), Wakelock():
        written = cache.build(trunk.slice_features, corpus, identifiers, device, batch=arguments.batch)
    logger.info(f"MEDICALNET EMBEDDINGS BUILD COMPLETE {written.features.shape} width={written.width}")


if __name__ == "__main__":
    main()
