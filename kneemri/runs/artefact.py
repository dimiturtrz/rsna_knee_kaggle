"""Every file one run leaves, named once: the only place a sibling's trailing spelling is written.

Each writer used to derive its own path from the checkpoint — `.partial`, `.resume`, `.holdout.npz`,
`.metrics.jsonl` — and each reader re-derived it, which is how `run.metrics.pt` and
`run.metrics.metrics.jsonl` both came to be written to disk by code that reached for `with_suffix`.
A name with a THIRD thing between stem and extension cannot be built by `with_suffix`; it has to be
built from the run's STEM, and the stem has to be recovered from whichever sibling a caller holds.
"""

from enum import StrEnum
from pathlib import Path


class Artefact(StrEnum):
    """One kind of file a run writes beside its checkpoint, valued by its trailing spelling."""

    CHECKPOINT = ".pt"
    PARTIAL = ".partial"
    RESUME = ".resume"
    HOLDOUT = ".holdout.npz"
    METRICS = ".metrics.jsonl"
    STEPS = ".steps.jsonl"
    """One row per optimiser step: loss, pre-clip gradient norm, learning rate per group."""
    DRAWS = ".draws.npz"
    """Which studies the sampler drew, per epoch — the data the run actually descended on."""
    EPOCHS = ".epochs.npz"
    """Every epoch's holdout predictions, where `HOLDOUT` keeps only the best one's."""

    @staticmethod
    def stem(path: Path) -> str:
        """The run's own name, with whichever sibling spelling `path` carries removed.

        Longest spelling first, so `.metrics.jsonl` is never read as a stem ending `.metrics` with a
        `.jsonl` extension. A path carrying none of them is a bare stem already.
        """
        name = path.name
        spelled = next((kind for kind in sorted(Artefact, key=len, reverse=True) if name.endswith(kind)), None)
        return name.removesuffix(spelled) if spelled else name

    def beside(self, path: Path) -> Path:
        """This kind's file for the run `path` belongs to — any sibling, or the bare stem, will do."""
        return path.with_name(Artefact.stem(path) + self)

    def aside(self, path: Path, run: int) -> Path:
        """Where an EARLIER run's file of this kind is kept once a re-run takes the name.

        Under `_run<n>`, the spelling a human reached for by hand the night it was first needed, so a
        rotated run reads as an ordinary run to every tool that globs a kind — except the metrics
        record, which has always rotated to `.metrics.<n>.jsonl` and is read by that name on disk.
        """
        stem = Artefact.stem(path)
        if self is Artefact.METRICS:
            return path.with_name(f"{stem}.metrics.{run}.jsonl")
        return path.with_name(f"{stem}_run{run}{self}")
