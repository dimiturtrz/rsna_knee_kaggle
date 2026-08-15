"""Probabilities already computed for a member on a fixed holdout — the store, not the reading.

Its own module because it is its own subject: `holdout_predictions` answers what a member says about
these studies, and this answers whether anyone already asked.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float

from kneemri.inference.members import Window
from kneemri.inference.test_time import SpatialViews

DIGEST = 16
"""Characters of the key's hash kept in a filename — 64 bits, against a few hundred cached readings."""


class PredictionCache:
    """Probabilities already computed for a member on a fixed holdout, keyed by everything that moves them.

    The docstring above says this class needs "a GPU, a cache and 90 seconds"; the cache in that sentence
    did not exist, and its absence is why the ANALYSIS half of the project was gated on a GPU it does not
    need. Every paired interval, oracle and subset search is numpy over arrays of shape (studies, 12) —
    but reaching those arrays meant re-reading every member through its encoder first, so none of it could
    run while the card was busy, which is a normal state rather than an exception.

    THE KEY IS THE WHOLE DESIGN. A cache that answers with the wrong array is worse than no cache, and
    this project's recurring defect — a value travels per-run in a config while a consumer reads a
    constant — is exactly what a too-short key would reproduce, silently, in the tool that chooses what to
    submit. So the key carries the checkpoint's identity INCLUDING size and mtime (a member retrained
    under the same name must MISS), the identifiers of the studies actually scored (a changed split
    misses), and the full reading policy — window starts, width, read size, view set, precision.
    """

    def __init__(self, directory: Path, studies: list[str], views: SpatialViews, *, refresh: bool = False) -> None:
        self.directory = directory
        self.studies = studies
        self.views = views
        # A cache told to ignore what it holds. The key should make a stale hit impossible; this is the
        # escape hatch for the day it is not, and it still WRITES, so the recomputation is kept.
        self.refresh = refresh

    def key(self, checkpoint: Path, window: Window, precision: torch.dtype) -> str:
        """The canonical description of one reading — written beside the array, not just hashed into it.

        Kept as readable JSON rather than a bare digest so a cached row can be audited after the fact:
        a hash tells you two runs disagreed, the parts tell you WHICH part disagreed.
        """
        stat = checkpoint.stat()
        return json.dumps(
            {
                "checkpoint": checkpoint.name,
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "studies": hashlib.sha256("\n".join(self.studies).encode()).hexdigest(),
                "count": len(self.studies),
                "window": {"starts": list(window.starts), "width": window.width, "size": window.size},
                "views": repr(self.views),
                "precision": str(precision),
            },
            sort_keys=True,
        )

    def read(self, key: str) -> Float[np.ndarray, "studies targets"] | None:
        """The stored probabilities for this exact reading, or None if it was never computed.

        Verifies the sidecar matches rather than trusting the filename: a digest collision, or a file
        copied between machines, would otherwise hand back another member's answers. A missing sidecar
        is a MISS rather than an error, because the pass that would have written it may have been killed
        — and this store's whole purpose is to survive a walk that gets interrupted.
        """
        stored = self._path(key)
        sidecar = stored.with_suffix(".json")
        if self.refresh or not stored.exists() or not sidecar.exists():
            return None
        if sidecar.read_text(encoding="utf-8") != key:
            return None
        return np.load(stored)

    def write(self, key: str, values: Float[np.ndarray, "studies targets"]) -> None:
        """Store one reading's probabilities together with the parts that identify it.

        THE ARRAY LANDS LAST, by rename. Written in place it was the two-step commit that a kill lands
        in the middle of: a killed walk left a complete `.npy` with no sidecar beside it, and every
        later read of that key then died on the sidecar instead of recomputing — a cache that turns an
        interruption into a permanent failure. A member costs hours to re-derive on a CPU, so the walk
        being killable is a property this store has to keep. Now the array's presence IS the commit.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        stored = self._path(key)
        pending = stored.with_suffix(".pending")
        # Through a handle, because `np.save` appends `.npy` to any path that does not already end in
        # it — a pending file named by suffix alone would land as `<digest>.pending.npy` and never be
        # renamed, leaving the walk with a directory of debris and no cache at all.
        with pending.open("wb") as handle:
            np.save(handle, values)
        stored.with_suffix(".json").write_text(key, encoding="utf-8")
        pending.replace(stored)

    def _path(self, key: str) -> Path:
        return self.directory / f"{hashlib.sha256(key.encode()).hexdigest()[:DIGEST]}.npy"
