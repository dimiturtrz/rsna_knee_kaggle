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
only if the ORDERING survives, which is what `--correlate` measures and why it is the first thing to
run: it reruns the pairs in `experiments/head_axes.jsonl` — head-side variants whose FINE-TUNED scores
are already recorded — and reports whether frozen puts them in the same order. It scores only the pairs
the fine-tuned ruler RESOLVES, so an unread pair the proxy happens to match cannot certify it. Today
that record holds exactly one resolved pair, so the honest reading is that this harness has not yet
been shown to rank anything.

THE VARIANTS ARE THE POINT, and until 2026-08-16 there were none: `pooled` hard-coded the elementwise
maximum, the head was always the attended one, and `anatomical_prior` was a FIELD `run` never read —
so three of the axes the docstring above advertises could not be expressed at all, and the fourth
would have been recorded in a config while the head was built from a constant. That is this project's
plumbing failure in its purest form: a knob that lands in the record and never reaches the consumer.
`TrialHead` is now built from the same registries the pixel model uses, so a variant asked for here is
the variant that runs.
"""

import argparse
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypedDict, override

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor, nn

from kneemri.corpus.anatomy import ANATOMY, SlotAffinity
from kneemri.corpus.cache_flags import CacheFlags
from kneemri.embeddings.embedding_cache import EmbeddingCache, EmbeddingSpec
from kneemri.evaluation.scores import LabelScores
from kneemri.findings import TARGETS
from kneemri.labels.targets import Targets
from kneemri.models.model import CLS_MEAN, POOL_PARTS
from kneemri.models.slice_pooling import MAXIMUM, SLICE_POOLINGS
from kneemri.models.slot_head import ATTENDED, SLOT_POOLINGS
from kneemri.training.objective import Objective, Predicted, RankedObjective

logger = logging.getLogger(__name__)

BATCH = 64
"""Larger than the pixel trainer's 12: nothing here holds an encoder's activations, and the ranking
term scores pairs WITHIN a batch, so a wider one sees quadratically more of them."""

HEAD_LR = 1e-3
"""The pixel trainer's head rate, unchanged — this is the same head on the same features."""

TRIAL_SLICE_POOLINGS = tuple(name for name, pooling in SLICE_POOLINGS.items() if pooling.per_slice)
"""The slice reductions expressible over CACHED features, which is not every one the model has.

DERIVED from the model's own registry rather than listed here, because a hand-written list is a second
place to add a pooling to and the trial's parser would have gone on rejecting one added to the model.
`per_slice` is exactly the property that decides it: a pooling that encodes each slice alone is a
function of the cached per-slice features, and `channels` — which stacks three slices into an image's
colour channels — happens BEFORE the encoder and changes the pixels it reads. Cached features cannot
express that, and offering it here would answer a different question under the same name.
"""


FINE_TUNED_FLOOR = 0.0062
"""The three-seed spread on the fine-tuned ruler — below it a pair is UNREAD, not equal.

`interpretations/kneemri/2026-08-11_what-actually-moved-the-number.md` measured it, and the correlation
scores only pairs above it. A proxy that agrees with an unresolved pair has agreed with noise, and
counting that as concordance is how a harness certifies itself.
"""

HEAD_AXES = Path("experiments/head_axes.jsonl")
"""The committed record of head-side pairs whose FINE-TUNED numbers are known."""


class TrialSettings(TypedDict, total=False):
    """The trial fields a recorded pair may vary — the JSON boundary, named rather than left open.

    Total is off because a pair states only the axis it moves; everything else stays at the trial's own
    value, which is what makes the two sides differ in exactly one thing.
    """

    slice_pooling: str
    slot_pooling: str
    anatomical_prior: bool
    rank_weight: float


@dataclass(frozen=True)
class HeadAxis:
    """One head-side pair, as the pixel arms scored it — the thing the proxy is checked against.

    `control` and `variant` name `HeadTrial` fields, so a record that names something the trial does not
    have raises out of `replace` rather than running a control against itself.
    """

    axis: str
    control: TrialSettings
    variant: TrialSettings
    fine_tuned_control: float
    fine_tuned_variant: float
    source: str

    @property
    def fine_tuned(self) -> float:
        """What the pixel arms said this axis is worth."""
        return self.fine_tuned_variant - self.fine_tuned_control

    def resolved(self, floor: float = FINE_TUNED_FLOOR) -> bool:
        """Whether the ruler that produced this pair could read it at all."""
        return abs(self.fine_tuned) >= floor

    @classmethod
    def read(cls, record: Path = HEAD_AXES) -> tuple["HeadAxis", ...]:
        """The pairs in the committed record, skipping its `note` lines.

        Notes live in the record rather than beside it so the caveat on a pair travels with the pair;
        a reader that choked on them would push the prose somewhere it can go stale separately.
        """
        import json  # noqa: PLC0415 - one call site, and this module is imported for its trainer

        lines = (json.loads(line) for line in record.read_text(encoding="utf-8").splitlines() if line.strip())
        return tuple(
            cls(
                axis=entry["axis"],
                control=entry["control"],
                variant=entry["variant"],
                fine_tuned_control=entry["fine_tuned_control"],
                fine_tuned_variant=entry["fine_tuned_variant"],
                source=entry["source"],
            )
            for entry in lines
            if "axis" in entry
        )


class TrialHead(nn.Module):
    """SliceAttention (optional) then a slot head, over cached per-slice features.

    The pixel model's own two stages with the encoder removed, built from the SAME registries so a
    variant named here is the variant the pixel trainer would build. Constructing it independently is
    how the two drift into answering different questions while sharing a flag name.
    """

    def __init__(
        self,
        dim: int,
        n_slot: int,
        slice_pooling: str = MAXIMUM,
        slot_pooling: str = ATTENDED,
        *,
        anatomical_prior: bool = False,
    ) -> None:
        super().__init__()
        if slice_pooling not in TRIAL_SLICE_POOLINGS:
            message = f"{slice_pooling!r} is not expressible over cached features: {TRIAL_SLICE_POOLINGS}"
            raise ValueError(message)
        prior = SlotAffinity(ANATOMY).requested(TARGETS, n_slot, wanted=anatomical_prior)
        aggregator = SLICE_POOLINGS[slice_pooling].aggregator
        self.slices = aggregator(dim) if aggregator is not None else None
        self.head = SLOT_POOLINGS[slot_pooling](dim, n_slot, len(TARGETS), prior)

    @override
    def forward(self, features: Float[Tensor, "b s g d"], mask: Float[Tensor, "b s"]) -> Float[Tensor, "b o"]:
        """One logit per diagnosis, from a study's per-slice features and which slots it has.

        Absent slots are zeroed rather than left as encoder output on blank pixels: the head's softmax
        reads the mask, and a stale vector behind a zero mask is a value waiting to be used by accident.
        """
        slices = self.slices
        pooled = slices(features) if slices is not None else features.amax(dim=2)
        return self.head(pooled * mask.unsqueeze(-1), mask)


@dataclass(frozen=True)
class HeadTrial:
    """One head fitted on frozen features, and what it scored.

    Frozen by construction: there is no encoder here to unfreeze, which is exactly why an epoch costs
    seconds and exactly why the result is a proxy.
    """

    epochs: int = 30
    rank_weight: float = 1.0
    slice_pooling: str = MAXIMUM
    slot_pooling: str = ATTENDED
    anatomical_prior: bool = False
    seed: int = 2026

    def features(self, features: NDArray[np.float16]) -> Float[Tensor, "b s g d"]:
        """The cached corpus as one float32 tensor, per-slice axis intact.

        Intact because the slice reduction is now a VARIANT: attention over slices is parameterised and
        learns during the fit, so a corpus reduced up front can only ever express the maximum.
        """
        return torch.from_numpy(np.ascontiguousarray(features)).float()

    def run(
        self,
        features: Float[Tensor, "b s g d"],
        mask: Float[Tensor, "b s"],
        targets: Targets,
        split: tuple[NDArray[np.int64], NDArray[np.int64]],
    ) -> float:
        """Fit on the training studies, return the best holdout mean AUC over the epochs."""
        torch.manual_seed(self.seed)
        training, holdout = split
        head = TrialHead(
            features.shape[-1],
            features.shape[1],
            self.slice_pooling,
            self.slot_pooling,
            anatomical_prior=self.anatomical_prior,
        )
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
                    Predicted(head(features[batch], mask[batch]), torch.from_numpy(rows), torch.from_numpy(weights))
                )
                loss.backward()
                optimiser.step()
            head.eval()
            with torch.no_grad():
                predicted = torch.sigmoid(head(features[holdout], mask[holdout])).numpy().astype(np.float64)
            best = max(best, scores.mean(predicted, targets.scores[holdout]))
        return best

    def _batches(self, studies: NDArray[np.int64], generator: np.random.Generator) -> list[NDArray[np.int64]]:
        """Shuffled batches of study positions, dropping the partial last one.

        Dropped for the same reason the pixel trainer drops it: the ranking term scores pairs within a
        batch, so a short final batch takes a step on a different number of pairs than every other.
        """
        order = generator.permutation(studies)
        return [order[start : start + BATCH] for start in range(0, len(order) - BATCH + 1, BATCH)]


@dataclass(frozen=True)
class AxisConcordance:
    """Whether the frozen proxy ORDERS head variants the way the fine-tuned arms did.

    THE ONLY THING THAT MAKES THE PROXY USABLE. A frozen encoder is a different function from one whose
    blocks adapt, so a head that wins here need not win there — and the whole value of a six-minute
    encode plus seconds-long epochs rests on an ordering claim nobody had checked. This checks it, on
    the pairs the fine-tuned ruler can actually READ.
    """

    trial: HeadTrial = HeadTrial()

    def proxy(
        self,
        axis: HeadAxis,
        features: Float[Tensor, "b s g d"],
        mask: Float[Tensor, "b s"],
        targets: Targets,
        split: tuple[NDArray[np.int64], NDArray[np.int64]],
    ) -> float:
        """What this axis is worth on cached features: variant minus control, both fitted here."""
        variant = replace(self.trial, **axis.variant).run(features, mask, targets, split)
        control = replace(self.trial, **axis.control).run(features, mask, targets, split)
        return variant - control

    def report(
        self,
        features: Float[Tensor, "b s g d"],
        mask: Float[Tensor, "b s"],
        targets: Targets,
        split: tuple[NDArray[np.int64], NDArray[np.int64]],
        axes: tuple[HeadAxis, ...] = (),
    ) -> int:
        """Log each pair and return how many RESOLVED pairs the proxy ordered the same way.

        Unresolved pairs are logged and not counted, so the denominator is the number of pairs the
        fine-tuned ruler could read — not the number of lines in the record, which would let an
        unreadable pair the proxy happened to match inflate the agreement.
        """
        agreed, resolved = 0, 0
        for axis in axes or HeadAxis.read():
            proxy = self.proxy(axis, features, mask, targets, split)
            if not axis.resolved():
                logger.info(
                    "%s: fine-tuned %+.4f INSIDE the floor, proxy %+.4f — unread, not scored",
                    axis.axis,
                    axis.fine_tuned,
                    proxy,
                )
                continue
            resolved += 1
            same = (proxy > 0) == (axis.fine_tuned > 0)
            agreed += same
            logger.info(
                "%s: fine-tuned %+.4f, proxy %+.4f — %s",
                axis.axis,
                axis.fine_tuned,
                proxy,
                "same direction" if same else "OPPOSITE direction",
            )
        logger.info("%d of %d resolved pair(s) agree", agreed, resolved)
        if not resolved:
            logger.info("the record resolves nothing, so this proxy is not yet shown to rank anything")
        return agreed


def main() -> None:
    """Fit one head on a cached corpus, or sweep the rank weight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="small")
    CacheFlags.add_to(parser)
    parser.add_argument("--read-size", type=int, default=336)
    parser.add_argument("--pool", default=CLS_MEAN, choices=tuple(POOL_PARTS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--rank-weight", type=float, nargs="+", default=[1.0], help="one value, or several to sweep")
    parser.add_argument("--slice-pooling", default=MAXIMUM, choices=TRIAL_SLICE_POOLINGS)
    parser.add_argument("--slot-pooling", default=ATTENDED, choices=tuple(SLOT_POOLINGS))
    parser.add_argument("--anatomical-prior", action="store_true")
    parser.add_argument(
        "--correlate",
        action="store_true",
        help=f"rerun the pairs in {HEAD_AXES} and report whether the ordering survives freezing",
    )
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

    features = HeadTrial().features(corpus.features)
    mask = torch.from_numpy(corpus.mask)
    if arguments.correlate:
        AxisConcordance(HeadTrial(epochs=arguments.epochs)).report(features, mask, targets, split)
        return
    for weight in arguments.rank_weight:
        trial = HeadTrial(
            epochs=arguments.epochs,
            rank_weight=weight,
            slice_pooling=arguments.slice_pooling,
            slot_pooling=arguments.slot_pooling,
            anatomical_prior=arguments.anatomical_prior,
        )
        logger.info("rank_weight %.2f: best holdout mean AUC %.4f", weight, trial.run(features, mask, targets, split))


if __name__ == "__main__":
    main()
