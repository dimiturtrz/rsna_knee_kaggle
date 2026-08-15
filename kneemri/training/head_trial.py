"""Fit the HEAD alone on cached encoder features: `python -m kneemri.training.head_trial`.

The other half of `embedding_cache`, which until now was written by `build_embeddings` and read by
nobody — a whole corpus encoded and consumed by nothing, which is the recorded-but-unread shape this
project keeps finding, at cache scale.

THE ARITHMETIC that makes it worth having. Whole-stack fold 0 is 4,407 studies x 6 slots x 12 slices =
317k images, encoded ONCE at ~940 img/s in about six minutes. After that a head epoch is a few matmuls
over a 487 MB array: seconds, against 320 s for a full arm. Every question that changes neither the
pixels nor the encoder becomes nearly free to ask — slot-head design, the anatomical prior, rank
weight, weight floor, loss shape, slice pooling, report supervision weight.

IT IS A PROXY AND NOT THE MODEL, which is the whole risk. A frozen encoder is a different function
from one whose last blocks adapt, so a head that wins here need not win there. The harness earns trust
only if the ORDERING survives: run head variants whose fine-tuned scores we already know, frozen, and
check they rank the same. Until that is measured this answers a different question than the one we
ask — `--correlate` is the flag that measures it, and it is the first thing to run.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor

from kneemri.corpus.cache_flags import CacheFlags
from kneemri.embeddings.embedding_cache import EmbeddingCache, EmbeddingSpec
from kneemri.evaluation.scores import LabelScores
from kneemri.labels.targets import Targets
from kneemri.models.model import CLS_MEAN, POOL_PARTS, TARGETS
from kneemri.models.slot_head import SlotHead
from kneemri.training.objective import Objective, Predicted, RankedObjective

logger = logging.getLogger(__name__)

BATCH = 64
"""Larger than the pixel trainer's 12: nothing here holds an encoder's activations, and the ranking
term scores pairs WITHIN a batch, so a wider one sees quadratically more of them."""

HEAD_LR = 1e-3
"""The pixel trainer's head rate, unchanged — this is the same head on the same features."""


@dataclass(frozen=True)
class HeadTrial:
    """One head fitted on frozen features, and what it scored.

    Frozen by construction: there is no encoder here to unfreeze, which is exactly why an epoch costs
    seconds and exactly why the result is a proxy.
    """

    epochs: int = 30
    rank_weight: float = 1.0
    anatomical_prior: bool = False
    seed: int = 2026

    def pooled(self, features: NDArray[np.float16], mask: NDArray[np.float32]) -> Float[Tensor, "b s d"]:
        """Per-slice vectors reduced to one per slot by the elementwise maximum over slices.

        The same reduction the whole-stack model applies, done here over CACHED per-slice features —
        which is what makes a slice-pooling question askable without touching the encoder. Absent slots
        are zeroed rather than left as encoder output on blank pixels, because the head's softmax reads
        the mask and a stale vector behind a zero mask is a value waiting to be used by accident.
        """
        pooled = torch.from_numpy(np.ascontiguousarray(features)).float().amax(dim=2)
        return pooled * torch.from_numpy(mask).unsqueeze(-1)

    def run(
        self,
        pooled: Float[Tensor, "b s d"],
        mask: Float[Tensor, "b s"],
        targets: Targets,
        split: tuple[NDArray[np.int64], NDArray[np.int64]],
    ) -> float:
        """Fit on the training studies, return the best holdout mean AUC over the epochs."""
        torch.manual_seed(self.seed)
        training, holdout = split
        head = SlotHead(pooled.shape[-1], pooled.shape[1], len(TARGETS))
        optimiser = torch.optim.AdamW(head.parameters(), lr=HEAD_LR)
        objective: Objective = RankedObjective(self.rank_weight) if self.rank_weight else Objective()
        scores = LabelScores()
        generator = np.random.default_rng(self.seed)
        best = 0.0
        for _ in range(self.epochs):
            head.train()
            for batch in self._batches(training, generator):
                rows, weights = targets.rows(batch)
                optimiser.zero_grad(set_to_none=True)
                loss = objective(
                    Predicted(head(pooled[batch], mask[batch]), torch.from_numpy(rows), torch.from_numpy(weights))
                )
                loss.backward()
                optimiser.step()
            head.eval()
            with torch.no_grad():
                predicted = torch.sigmoid(head(pooled[holdout], mask[holdout])).numpy().astype(np.float64)
            best = max(best, scores.mean(predicted, targets.scores[holdout]))
        return best

    def _batches(self, studies: NDArray[np.int64], generator: np.random.Generator) -> list[NDArray[np.int64]]:
        """Shuffled batches of study positions, dropping the partial last one.

        Dropped for the same reason the pixel trainer drops it: the ranking term scores pairs within a
        batch, so a short final batch takes a step on a different number of pairs than every other.
        """
        order = generator.permutation(studies)
        return [order[start : start + BATCH] for start in range(0, len(order) - BATCH + 1, BATCH)]


def main() -> None:
    """Fit one head on a cached corpus, or sweep the rank weight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="small")
    CacheFlags.add_to(parser)
    parser.add_argument("--read-size", type=int, default=336)
    parser.add_argument("--pool", default=CLS_MEAN, choices=tuple(POOL_PARTS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--rank-weight", type=float, nargs="+", default=[1.0], help="one value, or several to sweep")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from core.paths import DataRoot  # noqa: PLC0415 - CLI-only, and this module is imported as a library
    from kneemri.corpus.folds import ReportFolds  # noqa: PLC0415 - as above

    root = DataRoot.from_config(arguments.paths)
    # The features are named after the PIXELS they were encoded from, so this asks the same question
    # `build_embeddings` answered rather than restating three of that cache's six axes.
    source = CacheFlags.cache(arguments, root.processed()).key
    spec = EmbeddingSpec(variant=arguments.variant, source=source, read_size=arguments.read_size, pool=arguments.pool)
    corpus, identifiers = EmbeddingCache(root.processed(), spec).load()
    targets = Targets.load(arguments.labels, root.raw() / "train.csv", identifiers)
    folds = ReportFolds(arguments.folds, arguments.split_seed)
    split = folds.holdout(folds.assign(str(root.raw() / "train.csv"), identifiers), arguments.fold)

    trial = HeadTrial(epochs=arguments.epochs)
    pooled = trial.pooled(corpus.features, corpus.mask)
    mask = torch.from_numpy(corpus.mask)
    for weight in arguments.rank_weight:
        scored = HeadTrial(epochs=arguments.epochs, rank_weight=weight).run(pooled, mask, targets, split)
        logger.info("rank_weight %.2f: best holdout mean AUC %.4f", weight, scored)


if __name__ == "__main__":
    main()
