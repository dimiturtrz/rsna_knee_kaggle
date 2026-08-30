"""Members grouped by the cache each one names, so every member is read from its OWN pixels.

Two analysis CLIs — the ensemble curve and the ruler check — ask different questions of the same set of
checkpoints, and both had to solve the identical problem first: a member records the reading it was
fitted under, several members usually share one, and decoding a 35 GB cache twice because two of them
were listed out of order is minutes of wall-clock per run. Both had also got it wrong in the same way,
loading one cache named by CLI flags and scoring every member from it.

So the grouping lives here once. The order of the FIRST cache is the reference: any other cache that
indexes the corpus differently is refused, because its rows would line up against the wrong studies and
produce a wrong number rather than an error.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from core.data.corpus.study_cache import CachedCorpus, StudyCache
from kneemri.inference.holdout_slab import HoldoutSlab
from kneemri.inference.members import Member

logger = logging.getLogger(__name__)


class MemberReadings:
    """Every member with the cache it names, grouped so each cache is loaded once."""

    def __init__(self, members: list[Member], directory: Path) -> None:
        # SKIPPED RATHER THAN REFUSED, the way a member with no holdout record is skipped. A caller that
        # globs the processed directory hands us the SSL encoders too, and they have no reading to group
        # by; raising here killed a pool sweep before its first real member (knee-o5m2).
        self.skipped = [member for member in members if not member.reads_pixels]
        for member in self.skipped:
            logger.info("%s: no reading recorded, not a member — skipped", member.file.name)
        members = [member for member in members if member.reads_pixels]
        self.members = members
        self.caches: dict[str, StudyCache] = {}
        self.owners: dict[str, list[Member]] = {}
        for member in members:
            cache = member.cache(directory)
            self.caches.setdefault(cache.key, cache)
            self.owners.setdefault(cache.key, []).append(member)

    @property
    def keys(self) -> list[str]:
        """The distinct caches these members were fitted on, first-listed first."""
        return list(self.caches)

    def reference(self) -> tuple[CachedCorpus, list[str]]:
        """The first member's cache and its study order — the order every other one must agree with."""
        return self.caches[self.keys[0]].load()

    def slabs(
        self, holdout: NDArray[np.int64], identifiers: list[str], corpus: CachedCorpus
    ) -> Iterator[tuple[Member, HoldoutSlab, CachedCorpus]]:
        """Each member with the holdout of its own cache, one load and one slab per distinct cache.

        The reference corpus is passed in rather than reloaded so the caller's own `load` is not paid
        twice; every other cache is loaded here and checked against the reference order first.
        """
        for key in self.keys:
            held = corpus
            if key != self.keys[0]:
                held, order = self.caches[key].load()
                if order != identifiers:
                    message = f"cache {key} indexes studies in a different order — its rows would not line up"
                    raise ValueError(message)
            slab = HoldoutSlab(held, holdout)
            for member in self.owners[key]:
                yield member, slab, held
