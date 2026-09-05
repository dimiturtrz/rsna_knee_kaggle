"""Does a cell's label disagree with the labels of its NEAREST NEIGHBOURS in a frozen trunk's feature space?

`python -m kneemri.evaluation.neighbour_errors`

CONFIDENT LEARNING IS THE MEASURED FLOOR HERE, NOT THE CEILING. `label_errors` scores the reader's cells
from twenty published checkpoints' out-of-fold probabilities and gates them on the 58 radiologist-annotated
studies: separation AUC 0.7628 over all 140 disagreements, precision 0.720 at the gold budget, and recall
**0.129**. All three label treatments built on it are now measured and none pays — prune at full budget
costs 0.0158, prune by ranking costs, relabel at the net-benefit-optimal budget is flat — and the flat is
explained by that recall rather than by the corrector, because a treatment that reaches under a tenth of
the corpus's wrong cells cannot move a mean of twelve AUCs. The lane reopens on a better DETECTOR and on
nothing else.

WHY kNN OVER FEATURES IS THE REPLACEMENT WORTH BUILDING FIRST. AQuA (Goswami, Sanil, Choudhry, Srinivasan,
Udompanyawit and Dubrawski, NeurIPS 2023 Datasets and Benchmarks, arXiv:2306.09467) benchmarks the label-error
families over seventeen datasets and four noise types and finds confident learning consistently the WORST of
the methods evaluated, with SimiFeat the best. SimiFeat (Zhu, Dong and Liu, ICML 2022, arXiv:2110.06283) is
training-free: it asks whether a sample's k nearest neighbours in a pretrained embedding carry the same label,
and on HUMAN annotation noise — the regime a report-mined table is in — it beats confident learning by about
twenty points of F1. Training-free is what makes it affordable here: the processed root already holds
full-corpus 336 px feature arrays for six frozen trunks, so the strongest published detector costs no card
and no submission slot.

THE TRUNKS ARE FROZEN AND WERE NEVER FITTED ON THESE LABELS, WHICH IS THE WHOLE POINT. `EmbeddingCache.key`
names a variant, a pixel source, a read size and two poolings — no checkpoint — because the features are the
pretrained encoder's, not a member's. A detector built from a member's own features would agree with the
labels it is auditing for the same reason `label_errors` refuses in-sample probabilities.

SIMIFEAT IS SINGLE-LABEL AND WE ARE NOT, AND THE COMPROMISE IS STATED RATHER THAN HIDDEN. Our defect has a
CELL as its unit — a (study, finding) pair — while a study carries ONE embedding and twelve labels. So the
neighbour set is shared across the twelve findings and only the label vector differs: this runs the vote
twelve times over one neighbourhood, which is the cheapest honest reading and also the most correlated one.
A globally atypical study is therefore liable to be flagged on several findings at once, and that is a known
cost of this design, not a property of the corpus. The gate below is what prices it — the same gate, the same
census, the same budgets `label_errors` is read at, so the two detectors are directly comparable and neither
is scored against a truth the other does not share.

THE VOTE IS SIMILARITY-WEIGHTED, WHICH IS NOT A KNOB BUT A RESOLUTION FIX. A plain majority over k
neighbours takes only k+1 distinct values, and a ranking with eleven levels cannot separate 684 cells; the
separation AUC would be measuring the tie-breaking rule. Weighting each neighbour by its cosine similarity
leaves the same ordering intent with a continuous score.

MULTIPLE TRUNKS ARE THIS METHOD'S TEST-TIME AUGMENTATION. SimiFeat aggregates over augmentation rounds to
stabilise a local vote; six independently pretrained encoders give the same stabilisation over a stronger
axis than an image transform does, and averaging their agreements is the aggregation the paper prescribes.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Bool, Float

from core import cli
from core.data.labels.reader_agreement import ReaderAgreement
from core.findings import TARGETS
from kneemri.evaluation.label_errors import DEFAULT_LABELS, LabelErrorAudit, ReaderTable

logger = logging.getLogger(__name__)

DEFAULT_VIEWS: tuple[str, ...] = (
    "ortholarge_336px_12sl_r256_cls_mean_focal_max",
    "v3large_336px_12sl_r256_mean_max",
    "large_336px_12sl_r252_mean_max",
    "small_336px_12sl_r336_cls_mean_max",
    "radnet121_336px_12sl_r256_cls_mean_focal_max",
    "vjepa2l_336px_12sl_r256_mean_channels",
)
"""One cache key per frozen trunk whose full-corpus 336 px features are already on disk.

Six encoders rather than the best one: the vote is local and a single embedding's neighbourhood is the one
thing this method cannot check for itself.
"""

DEFAULT_NEIGHBOURS = 10
"""SimiFeat's own k (arXiv:2110.06283). Large enough for a vote to survive one atypical neighbour, small
enough that the neighbourhood is still local at 4,407 studies."""

MAJORITY = 0.5
"""The method's OWN operating point — a weighted majority of the neighbours disagreeing with the label.

Named because it is the counterpart of cleanlab's confident-joint mask: a flagged-set size the detector
chooses rather than one this project picks to flatter a budget curve.
"""

NEIGHBOUR_VOTE = "neighbour vote:"
CONFIDENT_LEARNING = "confident learning, the measured floor:"
COMBINED = "rank-average of the two:"
"""Section headings, because three gates rendered one after another with no name is three numbers nobody
can attribute — the same defect `mask_name` fixes one line lower down."""

NEIGHBOUR_MASK = "the vote's own majority"
COMBINED_MASK = "both masks agreeing"
"""The combination has no self-chosen operating point of its own, so it borrows the one place both detectors
committed: a cell each flags. An intersection is the honest reading of two masks neither of which is a
threshold this project picked."""

CHUNK = 256
"""Studies reduced per block. The caches are up to 1.9 GB of float16 and are read through a memory map."""

EPSILON = 1e-12


@dataclass(frozen=True)
class TrunkView:
    """One frozen trunk's corpus embedding, reduced to a unit vector per study.

    BESIDE ITS CONSUMER because the reduction is the decision: a study is stored as slots by slices by
    channels, and which of those axes are averaged away is what makes two studies comparable at all.
    """

    identifiers: list[str]
    unit: Float[np.ndarray, "studies dim"]
    """L2-normalised, so the neighbour search is cosine. Trunk feature magnitudes differ by about 17x
    across this project's encoders (knee-x4yx), and an unnormalised vote would rank by trunk scale."""

    @classmethod
    def of(cls, directory: Path, key: str) -> "TrunkView":
        """Load one embedding cache and reduce it to a study-level unit vector."""
        identifiers = json.loads((directory / f"features_index_{key}.json").read_text())
        features = np.load(directory / f"features_{key}.npy", mmap_mode="r")
        present = np.load(directory / f"features_mask_{key}.npy")
        return cls(list(identifiers), cls._unit(cls._pooled(features, present)))

    @staticmethod
    def _pooled(
        features: Float[np.ndarray, "studies slots slices dim"], present: Bool[np.ndarray, "studies slots"]
    ) -> Float[np.ndarray, "studies dim"]:
        """Mean over the slices inside a slot, then over the slots the study actually carries.

        Weighted by the presence mask rather than averaged flat: a study missing a sequence would otherwise
        be pulled toward the origin by its zero rows, and distance to the origin is not similarity.
        """
        studies, _, _, dim = features.shape
        weight = present.astype(np.float32)
        pooled = np.zeros((studies, dim), dtype=np.float32)
        for start in range(0, studies, CHUNK):
            stop = min(start + CHUNK, studies)
            slots = np.asarray(features[start:stop], dtype=np.float32).mean(axis=2)
            share = weight[start:stop, :, None]
            pooled[start:stop] = (slots * share).sum(axis=1) / np.maximum(share.sum(axis=1), EPSILON)
        return pooled

    @classmethod
    def loaded(cls, directory: Path, keys: tuple[str, ...]) -> tuple["TrunkView", ...]:
        """Load every named cache, skipping the ones this machine does not hold."""
        loaded: list[TrunkView] = []
        for key in keys:
            if not (directory / f"features_{key}.npy").exists():
                logger.warning(f"no cache for {key}, skipping this view")
                continue
            loaded.append(cls.of(directory, key))
            logger.info(f"loaded {key}")
        if not loaded:
            message = f"none of the {len(keys)} named feature caches are on this machine"
            raise FileNotFoundError(message)
        return tuple(loaded)

    @staticmethod
    def aligned(views: tuple["TrunkView", ...]) -> list[str]:
        """The study order every view shares, refusing rather than joining if they disagree.

        A silent reindex here would pair one trunk's study with another trunk's label, which is a detector
        that runs, reports, and means nothing.
        """
        identifiers = views[0].identifiers
        for view in views[1:]:
            if view.identifiers != identifiers:
                message = "the feature caches do not share a study order"
                raise ValueError(message)
        return identifiers

    @staticmethod
    def _unit(pooled: Float[np.ndarray, "studies dim"]) -> Float[np.ndarray, "studies dim"]:
        """The same vectors on the unit sphere, so an inner product IS the cosine."""
        return pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), EPSILON)


@dataclass(frozen=True)
class NeighbourVote:
    """SimiFeat's local vote, run once per finding over a shared study neighbourhood.

    Holds the views and the reader together rather than taking them per call: the agreement matrix and the
    dissent mask are two readings of ONE vote, and a caller that built them from separate runs is how a
    flagged set stops describing the ranking it is quoted beside.
    """

    views: tuple[TrunkView, ...]
    reader: ReaderTable
    neighbours: int = DEFAULT_NEIGHBOURS

    def agreement(self) -> Float[np.ndarray, "studies targets"]:
        """Per-cell similarity-weighted share of the k nearest neighbours carrying the SAME label.

        LOW IS SUSPECT, which is `LabelErrorDetector.quality`'s convention exactly — the gate sorts
        ascending and a detector that inverted the sign here would read as worse than chance rather than
        as reversed. Unread cells are NaN, never 0, for the same reason they are there: a zero would rank
        every abstention above every real disagreement.

        A plain mean rather than a NaN-tolerant one: every view votes over the SAME reader, so the unread
        cells are the same cells in all of them, and a view that had somehow scored a cell the others left
        unread would be a bug worth propagating rather than one worth averaging away.
        """
        return np.mean(np.stack([self._voted(view) for view in self.views]), axis=0)

    def dissent(self) -> Bool[np.ndarray, "studies targets"]:
        """The cells a weighted MAJORITY of the neighbourhood contradicts — the method's own operating point.

        The counterpart of cleanlab's confident-joint mask: the size of this set is the detector's estimate
        rather than a budget chosen here.
        """
        agreement = self.agreement()
        return ~np.isnan(agreement) & (agreement < MAJORITY)

    def _voted(self, view: TrunkView) -> Float[np.ndarray, "studies targets"]:
        """One trunk's vote for every finding, over the rows that finding is read on."""
        similarity = view.unit @ view.unit.T
        np.fill_diagonal(similarity, -np.inf)
        scores = np.full((len(view.identifiers), len(TARGETS)), np.nan, dtype=np.float64)
        for column in range(len(TARGETS)):
            rows = np.flatnonzero(self.reader.read[:, column])
            if rows.size <= self.neighbours:
                continue
            scores[rows, column] = self._column(similarity[np.ix_(rows, rows)], self.reader.given[rows, column])
        return scores

    def _column(
        self, local: Float[np.ndarray, "rows rows"], labels: Bool[np.ndarray, "rows"]
    ) -> Float[np.ndarray, "rows"]:
        """The weighted agreement of each row's k nearest neighbours WITHIN one finding's read rows.

        The self-similarity is already -inf, so a row can never vote for itself; negative cosines are
        clipped away rather than allowed to subtract, because a dissimilar study's label is evidence about
        nothing and a negative weight would make it evidence for the opposite.
        """
        nearest = np.argpartition(-local, self.neighbours - 1, axis=1)[:, : self.neighbours]
        weight = np.maximum(np.take_along_axis(local, nearest, axis=1), 0.0)
        same = (labels[nearest] == labels[:, None]).astype(np.float64)
        return (weight * same).sum(axis=1) / np.maximum(weight.sum(axis=1), EPSILON)


@dataclass(frozen=True)
class CombinedRanking:
    """Two detectors read as one, averaged over their per-cell RANKS rather than their scores.

    RANKS BECAUSE THE SCALES ARE NOT THE SAME QUANTITY. A held-out self-confidence and a similarity-weighted
    vote share both live in [0, 1] and mean different things; averaging them directly would hand the decision
    to whichever happens to have the wider spread, which is a property of the estimator and not of the cell.

    WHY COMBINING IS THE RIGHT MOVE AND NOT A HEDGE. The two are measured to fail differently on this census:
    confident learning reads the misses at 0.6903 and the neighbour vote reads them BELOW chance, while the
    neighbour vote reads the confident false positives higher than confident learning does. Two rankings whose
    errors are that differently placed are the case where an average beats both, and the same gate prices it.
    """

    parts: tuple[Float[np.ndarray, "studies targets"], ...]

    def quality(self) -> Float[np.ndarray, "studies targets"]:
        """The mean of each part's normalised rank. LOW IS SUSPECT, as in every detector this gate reads."""
        return np.mean(np.stack([self._ranked(part) for part in self.parts]), axis=0)

    @staticmethod
    def onto(
        scores: Float[np.ndarray, "studies targets"], identifiers: list[str], target: list[str]
    ) -> Float[np.ndarray, "target_studies targets"]:
        """One detector's cells re-indexed onto another's study order, absent studies left unread.

        The feature caches and the published ruler cover the corpus in DIFFERENT orders and need not cover
        the same studies at all, so the two rankings are joined on the study identifier, never on the row.
        """
        row = {name: index for index, name in enumerate(identifiers)}
        placed = np.full((len(target), scores.shape[1]), np.nan, dtype=np.float64)
        known = [(index, row[name]) for index, name in enumerate(target) if name in row]
        placed[[index for index, _ in known]] = scores[[source for _, source in known]]
        return placed

    @staticmethod
    def _ranked(scores: Float[np.ndarray, "studies targets"]) -> Float[np.ndarray, "studies targets"]:
        """One detector's read cells as ranks in [0, 1], its unread cells left NaN.

        Ranked over the READ cells only. Including the abstentions would put every unread cell at one end of
        the scale and make the average a function of how much a detector declined to score.
        """
        flat = np.asarray(scores).reshape(-1)
        read = ~np.isnan(flat)
        ranks = np.full(flat.shape, np.nan, dtype=np.float64)
        order = np.argsort(np.argsort(flat[read], kind="mergesort"), kind="mergesort")
        ranks[read] = order / max(int(read.sum()) - 1, 1)
        return ranks.reshape(np.asarray(scores).shape)


def main() -> None:
    """Score the reader's cells by a neighbour vote, and gate it on the annotated 58."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--labels", type=Path, default=None, help="the reader table to audit")
    command.parser.add_argument("--views", type=str, default=None, help="comma-separated feature cache keys")
    command.parser.add_argument("--neighbours", type=int, default=DEFAULT_NEIGHBOURS, help="the vote's k")
    command.parser.add_argument(
        "--against-confident", action="store_true", help="also read confident learning, and the two combined"
    )
    arguments = command.parsed()

    root = command.rooted(arguments)
    processed = root.processed()
    labels = arguments.labels or processed / DEFAULT_LABELS
    keys = DEFAULT_VIEWS if arguments.views is None else tuple(arguments.views.split(","))
    logger.info(f"auditing {labels} against a {arguments.neighbours}-neighbour vote over {len(keys)} trunks")

    views = TrunkView.loaded(processed, keys)
    identifiers = TrunkView.aligned(views)
    table = pd.read_csv(labels)
    reader = ReaderTable.of(table, identifiers)
    vote = NeighbourVote(views, reader, arguments.neighbours)
    audit = LabelErrorAudit(table, ReaderAgreement.annotated(root), processed / "published")
    agreement, dissent = vote.agreement(), vote.dissent()
    lines = [NEIGHBOUR_VOTE, *audit.gated(identifiers, reader, agreement, dissent, NEIGHBOUR_MASK).lines()]
    if arguments.against_confident:
        found = audit.detected()
        placed = CombinedRanking.onto(agreement, identifiers, found.identifiers)
        voted = np.nan_to_num(CombinedRanking.onto(dissent.astype(np.float64), identifiers, found.identifiers))
        combined = CombinedRanking((placed, found.quality)).quality()
        both = (voted > 0.0) & found.flagged
        lines += [
            CONFIDENT_LEARNING,
            *audit.gated(found.identifiers, found.reader, found.quality, found.flagged).lines(),
        ]
        lines += [COMBINED, *audit.gated(found.identifiers, found.reader, combined, both, COMBINED_MASK).lines()]
    command.report(lines)


if __name__ == "__main__":
    main()
