"""The command line every foreign corpus builder shares, around the one scheme it is allowed to write.

Rijeka, MRNet, KMAR and fastMRI each spelled out the same opening — the cache flags, `--out`, the scheme
guard — and the same close: metadata, the decode under a wakelock, the completion line. What differs is
the corpus and its own flags, which each builder adds between `opening` and `opened`.
"""

import argparse
import logging
from collections.abc import Callable, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from core import cli
from core.data.corpus.cache_flags import CacheFlags
from core.data.corpus.study_cache import StudyCache
from core.data.corpus.study_metadata import StudyMetadata
from core.paths import DataRoot
from core.wakelock import Wakelock


class Decodable(Protocol):
    """A foreign corpus as its builder drives it: the metadata it states, and the decode that fills the cache."""

    def metadata(self) -> list[StudyMetadata]: ...

    def build(self) -> None: ...


@dataclass(frozen=True)
class BuildSite:
    """What a foreign builder's command line resolved to: its arguments, its data root, the cache it named."""

    log: logging.Logger
    arguments: argparse.Namespace
    root: DataRoot
    cache: StudyCache


@dataclass(frozen=True)
class CorpusBuild:
    """One foreign corpus's build: the scheme it must write under, its name in the log, its metadata table."""

    scheme: str
    name: str
    metadata: str

    @staticmethod
    def opening(command: cli.Command) -> argparse.ArgumentParser:
        """Attach the cache flags and hand back the parser, for the builder to add its own source flags."""
        CacheFlags.add_to(command.parser)
        return command.parser

    def opened(self, command: cli.Command) -> BuildSite:
        """Parse the builder's argv and name its cache, refusing every scheme but this corpus's own.

        THE ONE ERROR THAT COSTS 71.6 GB. `Routing.suffix()` is empty for the canonical scheme, so a
        336px/12sl build under it writes to `images_336px_12sl.npy` — the arrays every member on disk was
        fitted on — and destroys them with a corpus differing only in its first axis. The flag defaults to
        canonical for every other CLI, so this refuses rather than assumes.
        """
        command.parser.add_argument(
            "--out", type=Path, default=None, help="cache directory (default: the processed root)"
        )
        arguments = command.parsed()
        if arguments.scheme != self.scheme:
            command.parser.error(f"--scheme {self.scheme} is required: any other scheme writes onto our own corpus")
        root = command.rooted(arguments)
        return BuildSite(command.log, arguments, root, CacheFlags.cache(arguments, arguments.out or root.processed()))

    def run(self, site: BuildSite, corpus: Decodable, rows: Callable[[], Sized], *, decode: bool = True) -> None:
        """Write the corpus's metadata, then decode it into the cache under a wakelock.

        The decode runs unattended for minutes and the idle timer counts INPUT rather than load (knee-wyzj).
        """
        StudyMetadata.write(corpus.metadata(), site.root.processed() / "metadata" / self.metadata)
        if not decode:
            site.log.info(f"{self.name} LABELS COMPLETE (no decode requested)")
            return
        site.log.info(f"building {site.cache.shape(len(rows()))} uint8 -> {site.cache.images_path}")
        with Wakelock():
            corpus.build()
        site.log.info(f"{self.name} BUILD COMPLETE")
