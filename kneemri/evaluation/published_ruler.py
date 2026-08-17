"""The whole-corpus ruler the twenty published checkpoints add up to.

Four seeds over five folds, whose `oof` id-sets partition all 4,407 studies exactly. Stacking them by
fold gives, for every study in the corpus, one prediction per seed from a model that did not train on
it — no labels, no leaderboard, no GPU.

WHY IT IS ITS OWN MODULE. It lived as a method on `OutOfFoldOracle`, which compares one published
member's recorded predictions against what our pipeline produces for the same studies. This compares
nothing: it is a RULER, and four evaluations (`paired_floor`, `ruler_transfer`, `stack_coverage`,
`supervision_gap`) were each constructing a residual-comparison oracle in order to reach it, three of
them through a local import to dodge the module's CLI-only dependencies. One class, two products.

WHY IT MATTERS. Our own ruler is one fold — 881 studies, 62 Fracture positives, a sampling noise floor
of 0.0053 that has decided what counts as a result for this whole project. This one is the entire
corpus: five times the studies, so about 0.0024. Every question that is arithmetic over member
predictions — per-label combining above all — can be asked here at more than twice the resolution.

WHAT IT CANNOT DO: it holds only THEIR members. Ours were never fitted fold-wise over the whole corpus,
so a comparison between ours and theirs still belongs on the shared fold-0 holdout. This ruler answers
questions about combining, not about whose weights are better.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass(frozen=True)
class PublishedFold:
    """One published checkpoint's out-of-fold block: which seed wrote it, and for which studies.

    BESIDE ITS PRODUCER rather than inlined into the stacking loop, because the two facts that make the
    stacking correct are facts about a single checkpoint — the studies identify the FOLD, and the seed
    identifies the column — and reading them off a tuple is how they get swapped.
    """

    seed: int
    identifiers: list[str]
    values: NDArray[np.float64]

    @classmethod
    def of(cls, checkpoint: Path) -> "PublishedFold":
        """Read one published `.pt`, taking only its oof block and its seed.

        `weights_only=False` because the payload is a config and two plain lists rather than tensors —
        and this reads files WE fetched from a named dataset and stored ourselves.
        """
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        return cls(
            seed=int(saved["config"]["seed"]),
            identifiers=list(saved["oof"]["ids"]),
            values=np.asarray(saved["oof"]["pred"], dtype=np.float64),
        )

    def fold(self) -> tuple[str, ...]:
        """What identifies the FOLD: the set of studies held out, order-independent.

        Two members of one fold hold the same studies and nothing promises they wrote them in the same
        order, so the sorted tuple is the only thing about a checkpoint that its fold-mates all agree on.
        """
        return tuple(sorted(self.identifiers))

    def aligned_to(self, order: list[str]) -> NDArray[np.float64]:
        """This fold's rows in someone else's order — REORDERED, never assumed.

        A row compared against the wrong study's answer is a residual no amount of staring at the reader
        will explain, and the two orders differ for no reason a caller can see.
        """
        position = {name: index for index, name in enumerate(self.identifiers)}
        return self.values[[position[name] for name in order]]


@dataclass(frozen=True)
class PublishedRuler:
    """Every study in the corpus, with one prediction per seed, held out."""

    published: Path
    """The directory the published checkpoints were unzipped into."""

    def per_seed(self) -> tuple[list[str], list[NDArray[np.float64]]]:
        """The corpus identifiers, and one array of predictions per seed aligned to them.

        A fold missing any seed is SKIPPED whole rather than padded: the arrays are returned as one
        block per seed, so a fold present for three of four seeds has no row to give the fourth, and
        the alternative is a ragged ruler whose columns mean different populations.

        Returned in exactly the shape `LabelCombine` and `MemberCurve` already take.
        """
        by_fold: dict[tuple[str, ...], list[PublishedFold]] = {}
        for checkpoint in sorted(self.published.glob("*.pt")):
            entry = PublishedFold.of(checkpoint)
            by_fold.setdefault(entry.fold(), []).append(entry)

        seeds = sorted({entry.seed for fold in by_fold.values() for entry in fold})
        names: list[str] = []
        stacked: list[list[NDArray[np.float64]]] = [[] for _ in seeds]
        for fold in by_fold.values():
            members = {entry.seed: entry for entry in fold}
            if set(members) != set(seeds):
                continue
            order = members[seeds[0]].identifiers
            names += order
            for column, seed in enumerate(seeds):
                stacked[column].append(members[seed].aligned_to(order))
        return names, [np.concatenate(rows) for rows in stacked]
