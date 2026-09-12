"""What our champions believe about a corpus nobody labelled: `python -m kneemri.inference.pseudo_labels`.

```
python -m kneemri.inference.pseudo_labels --scheme merged3 --image-size 224 --slices 12 \
    --identifiers <kmar index.json> --checkpoints a.pt b.pt --out labels/kmar_pseudo.csv
```

WHY THIS IS NOT `teacher_readings`. That module writes ORDER — normalised ranks, calibration discarded
on purpose — because its consumer is a distilling term that weights pairs by separation. This one writes
a LABELS TABLE, and the trainer reads a labels table as a level the loss pulls each cell toward. A rank
has no level, so the two cannot share a writer: rewriting these probabilities as ranks would hand every
corpus the same uniform target regardless of what the teachers actually believed about it.

FLOATS ARE ALREADY THE MECHANISM, so this needs no new machinery downstream. `rijeka_corpus` writes
`READER_LEVEL = (0.08, 0.82, 0.94)` straight into its ACL column and the trainer has consumed it since;
a soft target is that same column with more values in it.

DARK TEACHERS ARE ALLOWED HERE, and that is the one place in this project where they are. Every other
consumer of a `--fold -1` member has to refuse it, because a member that saw every holdout study leaks
the evaluation labels into whatever it teaches. A FOREIGN corpus is in no holdout of ours: these studies
are new patients from a different hospital, so there is nothing for a dark champion to leak and the
champion pool is simply the best teacher available. What it does NOT license is reading a score off the
student on OUR holdout and calling the gain a generalisation result — the student's teacher saw those
909 studies, so that number prices the distillation, not the new data.

THE CONFIDENCE COLUMN IS TEACHER AGREEMENT, not a constant. `report_labels_v2` uses `__conf` as a WEIGHT
(0.95 on YES, 0.85 on NO, 0.05 on the 27.3% of cells the report never named), so a pseudo-label whose
teachers disagree must pull less hard than one they agree on. The weight is `1 - 2 * sd` across the
teachers, and the 2 is DERIVED rather than chosen: a bounded [0, 1] variable attains its largest possible
standard deviation, 0.5, when its mass sits half at each end, so `1 - 2 * sd` covers exactly the unit
interval and needs no clip to stay inside it. A maximally split pool teaches at 0, unanimous teachers at
1, and two teachers at 0.1 and 0.9 keep 0.2 — weak, not silenced, which is what a pool that agrees on
the SIGN and not the level has earned. A constant 1.0 here would train the student hardest exactly where
the pool is least sure, which is the failure mode every self-training paper reports as confirmation bias.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.data.corpus.cache_flags import CacheFlags
from core.data.corpus.study_cache import CachedCorpus
from core.findings import TARGETS
from kneemri.inference.holdout_predictions import HoldoutPredictions
from kneemri.inference.holdout_slab import HoldoutSlab
from kneemri.inference.members import Member
from kneemri.models.backbone import ENCODERS

logger = logging.getLogger(__name__)

IDENTIFIER = "StudyInstanceUID"
CONFIDENCE = "__conf"
DISAGREEMENT = 2.0
"""The reciprocal of 0.5, the largest standard deviation a [0, 1] variable can have — so the weight spans
exactly the unit interval and a maximally split pool lands at zero rather than at a tuned constant."""


class PseudoLabels:
    """One soft labels table, written from a pool of teachers over pixels nobody labelled."""

    def __init__(self, predictions: HoldoutPredictions) -> None:
        self.predictions = predictions

    def _believed(self, member: Member, slab: HoldoutSlab) -> Float[NDArray[np.float64], "studies findings"]:
        """One teacher's probabilities over every study the slab serves, at the member's own window."""
        encoder = ENCODERS[member.variant]
        model = self.predictions.load(member.file, encoder.load(), encoder.dim, member.slot_count)
        return self.predictions.predict(model, slab, member.window(slab.depth, overlap=False), member.precision)

    @staticmethod
    def pooled(
        beliefs: list[Float[NDArray[np.float64], "studies findings"]],
    ) -> tuple[Float[NDArray[np.float64], "studies findings"], Float[NDArray[np.float64], "studies findings"]]:
        """The teachers' mean belief and the weight their agreement earns it.

        The MEAN of probabilities rather than of ranks or of logits: the consumer is a level, and a rank
        average has no level while a logit average moves the level with the pool's calibration rather
        than with its belief. One teacher is a legal pool and its spread is zero, so it teaches at full
        weight — which is correct, since there is no disagreement to discount.
        """
        stacked = np.stack(beliefs)
        spread = stacked.std(axis=0) if len(stacked) > 1 else np.zeros_like(stacked[0])
        return stacked.mean(axis=0), np.clip(1.0 - DISAGREEMENT * spread, 0.0, 1.0)

    @staticmethod
    def table(
        identifiers: list[str],
        levels: Float[NDArray[np.float64], "studies findings"],
        weights: Float[NDArray[np.float64], "studies findings"],
    ) -> pd.DataFrame:
        """The labels table in the layout `build_merged` unions and the trainer joins by identifier."""
        columns: dict[str, object] = {IDENTIFIER: identifiers}
        for index, target in enumerate(TARGETS):
            columns[target] = levels[:, index]
            columns[f"{target}{CONFIDENCE}"] = weights[:, index]
        return pd.DataFrame(columns)

    @staticmethod
    def rows(index: Path, offset: int) -> tuple[list[str], NDArray[np.int64]]:
        """The identifiers to label and where they sit in the cache being read.

        The two are separate arguments because the corpus that NAMES the studies and the cache that
        HOLDS them are allowed to differ: a merged cache appends a foreign corpus at an offset, and its
        own index is the union. Reading the foreign index alone and then indexing the merged array from
        zero would label our own studies with our own teachers and write the answer under someone
        else's identifiers.
        """
        identifiers = json.loads(index.read_text())
        return identifiers, np.arange(offset, offset + len(identifiers), dtype=np.int64)


def main() -> None:
    command = cli.Command(__doc__, logger)
    CacheFlags.add_to(command.parser)
    command.parser.add_argument("--identifiers", type=Path, required=True, help="the foreign cache's index.json")
    command.parser.add_argument("--offset", type=int, default=0, help="where those studies start in the cache read")
    command.parser.add_argument("--checkpoints", type=Path, nargs="+", required=True, help="the teachers")
    command.parser.add_argument("--out", type=Path, required=True, help="where the labels table is written")
    command.parser.add_argument("--device", default="cuda", help="the device the teachers serve on")
    arguments = command.parsed()

    root = command.rooted(arguments)
    cache = CacheFlags.cache(arguments, root.processed())
    identifiers, positions = PseudoLabels.rows(arguments.identifiers, arguments.offset)
    corpus = CachedCorpus(np.load(cache.images_path, mmap_mode="r"), np.load(cache.mask_path).astype(np.float32))
    slab = HoldoutSlab(corpus, positions)

    labels = PseudoLabels(HoldoutPredictions(torch.device(arguments.device)))
    beliefs = []
    for checkpoint in arguments.checkpoints:
        member = Member.from_checkpoint(checkpoint)
        logger.info(f"teaching from {checkpoint.stem} over {slab.count} studies at depth {slab.depth}")
        beliefs.append(labels._believed(member, slab))  # noqa: SLF001
    levels, weights = PseudoLabels.pooled(beliefs)

    out: Path = arguments.out
    out.parent.mkdir(parents=True, exist_ok=True)
    labels.table(identifiers, levels, weights).to_csv(out, index=False)
    logger.info(f"wrote {len(identifiers)} soft rows, mean weight {weights.mean():.4f} -> {out}")


if __name__ == "__main__":
    main()
