"""What a run predicted for EVERY study it held out, written beside the checkpoint that predicted it.

The training loop already computes this array: `Fitting.ranked` scores each epoch by running the whole
holdout through `Trainer.predict`, and the best epoch's output is what every per-label number in the
ledger was read off. Eight of those studies were then kept as a witness and the other 873 were dropped
on the floor — so the project could state a per-label delta but could not state its own interval, and
knee-3vh had to borrow a floor measured on someone else's seed twins.

Persisting it costs 42 KB and no compute, and the array is EXACT rather than a re-derivation: a second
inference pass over the same weights would land within a tolerance, and a tolerance is precisely what a
paired interval of +/-0.016 cannot afford.

THE TRUTH TRAVELS WITH THE PREDICTIONS, because two arms are only pairable if they were scored by the
SAME ruler. This project has three label tables and has already re-run its ablations after one of them
moved; a paired comparison across that boundary would be arithmetic on two different questions, and
`aligned` refuses it rather than averaging over it.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

SUFFIX = ".holdout.npz"

TWINS = (SUFFIX, ".metrics.jsonl")
"""The other names one run answers to. Named here rather than derived, so a new sibling file is a
one-line decision instead of a silent miss in whichever tool happens to hold that spelling."""

RULER_TOLERANCE = 1e-6
"""Truth is stored as float32 and compared after a round trip; anything wider is a different table."""


@dataclass(frozen=True)
class HoldoutReading:
    """One run's per-study probabilities, the labels they were scored against, and which epoch's."""

    studies: list[str]
    predictions: Float[NDArray[np.float64], "studies labels"]
    truth: Float[NDArray[np.float32], "studies labels"]
    epoch: int

    def aligned(
        self, other: "HoldoutReading"
    ) -> tuple[
        Float[NDArray[np.float64], "shared labels"],
        Float[NDArray[np.float64], "shared labels"],
        Float[NDArray[np.float32], "shared labels"],
    ]:
        """Both runs' predictions for the studies they SHARE, plus the one truth they agree on.

        Positions are not comparable across runs — a different fold, a different corpus build or a
        dropped study all reindex the array while leaving its shape plausible — so the join is by study
        NAME and the overlap is whatever survives it.

        Raises when the shared studies carry different labels, which means the two arms were scored by
        different rulers. That is the one failure a paired interval cannot detect for itself: the
        arithmetic succeeds, the number comes out tighter than either arm deserves, and it answers a
        question nobody asked.
        """
        mine = {name: position for position, name in enumerate(self.studies)}
        shared = [name for name in other.studies if name in mine]
        if not shared:
            message = f"no study is held out by both runs ({len(self.studies)} and {len(other.studies)} studies)"
            raise ValueError(message)
        here = np.array([mine[name] for name in shared], dtype=np.int64)
        there = np.array([position for position, name in enumerate(other.studies) if name in mine], dtype=np.int64)
        truth, theirs = self.truth[here], other.truth[there]
        if not np.allclose(truth, theirs, atol=RULER_TOLERANCE, equal_nan=True):
            message = f"the two runs were scored by different label tables over their {len(shared)} shared studies"
            raise ValueError(message)
        return self.predictions[here], other.predictions[there], truth


class HoldoutRecord:
    """The holdout predictions file beside a run's checkpoint — the store, not the reading."""

    def __init__(self, destination: Path) -> None:
        # ANY OF A RUN'S OWN NAMES REACHES ITS PREDICTIONS. The trainer passes its checkpoint, an
        # analysis passes the file it can see on disk, and `RunComparison` holds runs as their METRICS
        # files — under a plain `with_suffix` that last one asks for `.metrics.holdout.npz`, finds
        # nothing, and reports a run that predicted nothing. Which is a lie the caller cannot see.
        name = destination.name
        stripped = next((name[: -len(twin)] for twin in TWINS if name.endswith(twin)), None)
        self.path = destination.with_name(stripped + SUFFIX) if stripped else destination.with_suffix(SUFFIX)

    def exists(self) -> bool:
        """Whether this run left its per-study predictions. Older runs did not, and cannot be paired."""
        return self.path.exists()

    def write(
        self,
        identifiers: list[str],
        holdout: NDArray[np.int64],
        predictions: Float[NDArray[np.float64], "studies labels"],
        truth: Float[NDArray[np.float32], "studies labels"],
        epoch: int,
    ) -> None:
        """Record the kept epoch's predictions under the NAMES of the studies they belong to.

        Overwrites, because a later epoch that improves replaces the kept one and the file must say
        what the checkpoint beside it holds.
        """
        named = np.array([identifiers[position] for position in holdout])
        np.savez(
            self.path,
            studies=named,
            predictions=np.asarray(predictions, dtype=np.float32),
            truth=np.asarray(truth, dtype=np.float32),
            epoch=np.int64(epoch),
        )

    def read(self) -> HoldoutReading | None:
        """What this run predicted, or None for a run that finished before this was written."""
        if not self.exists():
            return None
        stored = np.load(self.path)
        return HoldoutReading(
            studies=[str(name) for name in stored["studies"]],
            predictions=stored["predictions"].astype(np.float64),
            truth=stored["truth"],
            epoch=int(stored["epoch"]),
        )
