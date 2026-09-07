"""Splitting studies so a holdout number means something.

Some reports are byte-identical across studies — a template read for an unremarkable knee. Measured on
our corpus: 4,276 distinct reports over 4,407 studies, so 177 studies sit in a duplicate group, the
largest being 37 studies sharing one Turkish all-normal template.

Every study in such a group receives the same derived target vector, so splitting a group across the
divide scores the model on a target whose source it trained on. Assigning by a hash of the report text
keeps each group whole; the leak is only 4% of studies, and removing it costs nothing.
"""

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

TRAIN_ONLY = -2
"""The fold of a study that may never be scored on — a corpus that is not the one we are measured on.

A merged cache holds our studies followed by a foreign corpus's, and hashing a foreign identifier the
ordinary way puts a fifth of them in the holdout. That silently changes what a holdout number MEANS: the
denominator becomes a different mix of corpora, and those rows carry labels for one finding out of
twelve, so the arm could not be read against the control it exists to be compared with.

NOT `-1`, which is already the sentinel for a run that reports no holdout at all: at `--fold -1` a
`-1` here would make the foreign corpus the ENTIRE holdout. `-2` is training under every fold, including
that one.

IT CHANGES NOTHING THAT EXISTS. Measured: 0 of our 4,407 studies are absent from `train.csv`, so before
a foreign corpus was on disk no study could reach this value.
"""

EXCLUDE_NONE = -1
"""The `excluded` fold that excludes nothing — no study is ever assigned it.

NOT `-2`, which every foreign study carries: excluding that value by default would silently drop the
merged corpus out of every run that uses one. `-1` is unassignable for the same reason `--fold -1`
holds nothing out, and it is the same sentinel on both sides for that reason.
"""


class ReportFolds:
    """Study-to-fold assignment by report-text hash, so duplicate reports never straddle the divide."""

    def __init__(self, folds: int = 5, seed: int = 2026) -> None:
        self.folds = folds
        self.seed = seed

    def assign(self, train_table: str, identifiers: list[str]) -> NDArray[np.int64]:
        """A fold index per study, stable across runs and machines.

        The hash is of the report text, not of the study id: two studies sharing a report must land in
        the same fold, which is the whole point.

        A study this table has never heard of is a study from another corpus, and it goes to `TRAIN_ONLY`
        rather than into the rotation — see that constant for why a holdout it appeared in would report a
        number nothing could be compared against.
        """
        reports = pd.read_csv(train_table).set_index("StudyInstanceUID")["Report"].fillna("").to_dict()
        assignments = np.empty(len(identifiers), dtype=np.int64)
        for index, identifier in enumerate(identifiers):
            if identifier not in reports:
                assignments[index] = TRAIN_ONLY
                continue
            digest = hashlib.sha1(f"{self.seed}:{reports[identifier]!s}".encode("utf-8")).hexdigest()
            assignments[index] = int(digest[:8], 16) % self.folds
        return assignments

    def holdout(
        self, assignments: NDArray[np.int64], fold: int, excluded: int = EXCLUDE_NONE
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Training and holdout positions for one fold — fixed rather than rotated within a run.

        `excluded` drops a SECOND fold from training without putting it in the holdout, which is what an
        out-of-fold teacher needs and `--fold` alone cannot express. A teacher fitted at `--fold k`
        trains on every other fold, fold 0 included; a student distilled from four such teachers and
        scored on fold 0 mimics four functions that each memorised fold 0's labels, and the fold-0
        number is inflated by that mimicry rather than by the treatment. Excluding fold 0 from every
        teacher's training set is the only thing that removes the path.
        """
        holdout = np.flatnonzero(assignments == fold)
        training = np.flatnonzero((assignments != fold) & (assignments != excluded))
        return training, holdout


@dataclass(frozen=True)
class FitPositions:
    """How much of a fold's split a run actually fits on: the smoke cap and the data-scaling fraction.

    Beside `ReportFolds` rather than on the options object these three flags are parsed into, because
    both operations take exactly the tuple `holdout` above returns and hand back the same tuple — they
    are the next two steps of the split, not a statement about a recipe. On `TrainingOptions` they were
    the only methods touching numpy and two of its six disjoint cohesion components; the flags stay
    scalar fields there, so the parser's dest names still map one-to-one onto that dataclass and every
    guard walking it flat keeps working.
    """

    studies: int = 0
    train_fraction: float = 1.0
    split_seed: int = 2026

    def sampled(
        self, training: NDArray[np.int64], holdout: NDArray[np.int64]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """The first `--studies` corpus positions, split in the ratio the fold already has.

        A SMOKE, not an arm: it proves the recipe WIRES — cache mapped, encoder fetched, model built,
        loss stepped, checkpoint written and reloaded — in a couple of minutes on the card rather than
        the forty a real fit costs. Every flag combination this repository ships has at some point been
        discovered miswired only after the GPU had already been held, which is the most expensive place
        to find out.

        The ratio is kept rather than a second flag added, so a smoke exercises BOTH halves of the loop:
        a run with a holdout of zero never scores, never selects a best epoch and so never proves the
        half of `Fitting` that writes the checkpoint. The score it produces is meaningless by
        construction — a few dozen studies — and `caveats` says so in the log for the same reason
        `--fold -1` says it cannot report one.
        """
        if not self.studies:
            return training, holdout
        share = len(training) / max(len(training) + len(holdout), 1)
        kept = min(max(round(self.studies * share), 1), len(training))
        return training[:kept], holdout[: max(self.studies - kept, 1)]

    def thinned(self, training: NDArray[np.int64]) -> NDArray[np.int64]:
        """A random `--train-fraction` share of the training studies, holdout untouched.

        DRAWN AT RANDOM AND NOT SLICED OFF THE FRONT, which `sampled` above is free to do because a
        smoke's score is meaningless anyway. Here it is the whole point: the corpus positions carry the
        order `ReportFolds` assigned them, so a prefix is a systematically chosen half and any
        difference it produces could be a property of that half rather than of its size.

        SEEDED BY `--split-seed`, not `--seed`, for the same reason the fold is: which studies exist is
        a property of the CORPUS and must not move when the model seed does. Two arms at the same
        fraction therefore fit the same studies, so a scaling curve and a recipe comparison can be run
        against each other without a second confound.

        The kept positions are re-sorted so that a thinned run reads the cache in the same ascending
        order a full one does — the loader's access pattern stays sequential rather than becoming a
        random gather over a 99 GB memory-map.
        """
        if self.train_fraction >= 1.0:
            return training
        kept = max(round(len(training) * self.train_fraction), 1)
        drawn = np.random.default_rng(self.split_seed).choice(len(training), size=kept, replace=False)
        return training[np.sort(drawn)]

    def caveats(self, training: NDArray[np.int64]) -> list[str]:
        """What the holdout this run reports is NOT, said where the positions were chosen.

        Beside the two rules rather than at the entrypoint, because both cuts produce a score that
        cannot be read against the ledger and only this object knows which cut was made. A run that
        thins or caps and says nothing files a number about a few dozen studies under an architecture.
        """
        lines = []
        if self.train_fraction < 1.0:
            lines.append(
                f"DATA-SCALING RUN: {self.train_fraction:.2f} of the training studies "
                f"({len(training)}), holdout untouched"
            )
        if self.studies:
            lines.append(f"SMOKE RUN: {self.studies} studies — the holdout it reports is not a result")
        return lines
