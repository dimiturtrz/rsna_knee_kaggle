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

from dataclasses import dataclass, replace
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

UNKNOWN_EPOCH = -1
"""The kept epoch, for a run that never recorded which one it was.

A NEGATIVE value rather than a plausible integer, because the plausible integer is what went wrong:
the back-fill used to fall back to `epochs_done`, which the older half of the ledger writes as the
run's CONFIGURED LENGTH — `wsfold0_e6` stamps 11 for weights that are the epoch-6 best, and three of
three pre-`epoch` checkpoints have `epochs_done == options.epochs`. The array is right either way, so
the stamp is cosmetic until something joins two records BY it, at which point a wrong-but-plausible
number is a silent mis-join and this one is a visible refusal."""

TRAINED = "trained"
"""The array the ledger's per-label numbers were read off, kept whole. Exact, by construction."""

DERIVED = "re-derived"
"""A second inference pass over the same weights, for a run that finished before the record existed.
It lands within the cross-precision tolerance (4x eps), which is WIDER than a paired half-width of
+/-0.016 — so an interval computed from one is indicative, and the file has to say so itself rather
than leave the caller to remember which runs were back-filled."""

FOREIGN = "foreign"
"""Someone else's weights, served through someone else's decode — `kneemri.evaluation.foreign_member`.

A third value rather than `DERIVED`, because the two differ in what a disagreement would MEAN. A
re-derived array is our weights read twice and differs from `TRAINED` only by precision. A foreign one
shares neither the trunk nor the feed, and the feed alone is worth -0.0348 on a byte-identical
checkpoint (0.9167 through their reader against 0.8819 through our best cache). A comparison that
cannot see which of the three it is holding will attribute that to weight quality."""

PROBE = "probe"
"""A ridge fitted on a trunk's CACHED features — `kneemri.evaluation.feature_probe --emit-holdout`.

A fourth value rather than `DERIVED`, because a probe is not the same trunk read twice: no layer of it
was fine-tuned, so its number is a property of the frozen representation and not of any checkpoint we
could ship. The distinction has already cost a lane once — `radnet121` probes 0.8197 and the same trunk
FITTED reads 0.6716, so a probe row that a tool mistook for a member's row would be off by 0.15 with
every gate green. Its epoch is `UNKNOWN_EPOCH`, which is the truth: a ridge has no epochs."""


@dataclass(frozen=True)
class HoldoutReading:
    """One run's per-study probabilities, the labels they were scored against, and which epoch's."""

    studies: list[str]
    predictions: Float[NDArray[np.float64], "studies labels"]
    truth: Float[NDArray[np.float32], "studies labels"]
    epoch: int
    source: str = TRAINED

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

    def at(self, studies: list[str]) -> "HoldoutReading":
        """This reading over `studies`, in THEIR order, joined by NAME.

        Positional alignment is the failure with no symptom: two members recorded over different
        holdouts line up perfectly under it and every number after is nonsense.
        """
        where = {study: position for position, study in enumerate(self.studies)}
        missing = [study for study in studies if study not in where]
        if missing:
            message = f"{len(missing)} studies are absent from the record ({missing[0]} first)"
            raise ValueError(message)
        rows = np.asarray([where[study] for study in studies], dtype=np.int64)
        return replace(self, studies=list(studies), predictions=self.predictions[rows], truth=self.truth[rows])


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

    @staticmethod
    def named(identifiers: list[str], holdout: NDArray[np.int64]) -> list[str]:
        """The holdout's study NAMES, in its own order — positions mean nothing outside one corpus build.

        Separate from `write` because a back-fill already holds names and only a training loop holds
        positions, and because the two spellings of "which studies" must not both be arguments to the
        same call: whoever passes them can pair the wrong ones, and the file would still load.
        """
        return [identifiers[position] for position in holdout]

    def write(
        self,
        studies: list[str],
        predictions: Float[NDArray[np.float64], "studies labels"],
        truth: Float[NDArray[np.float32], "studies labels"],
        epoch: int,
        source: str = TRAINED,
    ) -> None:
        """Record the kept epoch's predictions under the NAMES of the studies they belong to.

        Overwrites, because a later epoch that improves replaces the kept one and the file must say
        what the checkpoint beside it holds.

        `source` travels with the array because a back-filled run's numbers are a re-derivation and a
        trained run's are the array itself, and nothing downstream can tell them apart by looking.
        """
        # RENAMED INTO PLACE, never written in place: this file is written at the end of an epoch and
        # re-written by a back-fill, and a kill during either used to leave a truncated archive where a
        # complete record had been — which `read` cannot answer None to, so one interrupted write took
        # every later comparison of that run down with it. The rename is the commit.
        pending = self.path.with_suffix(".pending.npz")
        np.savez(
            pending,
            studies=np.array(studies),
            predictions=np.asarray(predictions, dtype=np.float32),
            truth=np.asarray(truth, dtype=np.float32),
            epoch=np.int64(epoch),
            source=np.array(source),
        )
        pending.replace(self.path)

    def restamp(self, epoch: int) -> bool:
        """Correct which epoch an existing record claims, without re-deriving it. Says whether it moved.

        The arrays are the expensive part and they were never wrong — a back-fill reads the weights on
        disk whatever epoch they came from. Only the stamp was guessed, so re-running 55 inference
        passes to repair one integer would cost hours and re-introduce the cross-precision tolerance
        into records that already sit inside it.
        """
        reading = self.read()
        if reading is None or reading.epoch == epoch:
            return False
        self.write(reading.studies, reading.predictions, reading.truth, epoch, source=reading.source)
        return True

    def read(self) -> HoldoutReading | None:
        """What this run predicted, or None for a run that finished before this was written.

        The two STAMPS are read defensively and the three ARRAYS are not, because they differ in what a
        missing key means. Six probe records on disk carry only `studies`, `predictions` and `truth` —
        they predate `write` and were emitted by a path that never had an epoch to state. Their stamps
        are exactly what `UNKNOWN_EPOCH` and `PROBE` were declared to say, so defaulting them is the
        vocabulary working rather than a guess. A missing ARRAY has no such reading: it is a truncated
        file, and it must still raise here instead of arriving downstream as an empty holdout.

        Read through `stored.files` rather than a try/except, so a record missing a stamp is a branch
        and not an exception path — `KeyError: 'epoch is not a file in the archive'` took the whole of
        `seat_value` down when one probe was named among four candidates (knee-ro2bu).
        """
        if not self.exists():
            return None
        stored = np.load(self.path)
        held = set(stored.files)
        return HoldoutReading(
            studies=[str(name) for name in stored["studies"]],
            predictions=stored["predictions"].astype(np.float64),
            truth=stored["truth"],
            epoch=int(stored["epoch"]) if "epoch" in held else UNKNOWN_EPOCH,
            source=str(stored["source"]) if "source" in held else PROBE,
        )

    def required(self) -> HoldoutReading:
        """What this run predicted, or a refusal naming the run that kept nothing."""
        reading = self.read()
        if reading is None:
            message = f"{self.path} left no holdout record - a --fold -1 member has none and can never have one"
            raise FileNotFoundError(message)
        return reading
