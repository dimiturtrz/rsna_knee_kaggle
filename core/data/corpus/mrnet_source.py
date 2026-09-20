"""Where the MRNet corpus's `.npy` volumes are read from — one interface over the two shapes it arrives in.

The Stanford download is a 5.7 GB zip and is read in place, because every array in it is used once and a
second copy on disk buys nothing. But a corpus that was extracted once is what is actually on this
machine, and `zipfile.ZipFile` on a directory raises `PermissionError: [Errno 13]` — a message about the
wrong thing entirely, at the end of a decode's launch rather than at its argument parse. Both shapes
answer the same two questions, so they are one type with two implementations rather than a branch at each
of `MRNetCorpus`'s three call sites.
"""

import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import override

VOLUME_SUFFIX = ".npy"
"""What a member of the corpus is called, in both shapes — the archive holds nothing else."""


class MRNetSource(ABC):
    """The corpus's members, named and read, however it happens to sit on disk."""

    @staticmethod
    def at(path: Path) -> "MRNetSource":
        """The reader the corpus at `path` needs, chosen by what it is rather than by what it is called."""
        return ExtractedSource(path) if path.is_dir() else ZippedSource(path)

    @abstractmethod
    def names(self) -> list[str]:
        """Every member, named the way `read` will take it back."""

    @abstractmethod
    def read(self, member: str) -> bytes:
        """One member's bytes."""


@dataclass(frozen=True)
class ZippedSource(MRNetSource):
    """The Stanford archive, read without extracting it."""

    archive: Path

    @override
    def names(self) -> list[str]:
        with zipfile.ZipFile(self.archive) as opened:
            return opened.namelist()

    @override
    def read(self, member: str) -> bytes:
        with zipfile.ZipFile(self.archive) as opened:
            return opened.read(member)


@dataclass(frozen=True)
class ExtractedSource(MRNetSource):
    """A corpus already unpacked on disk, named by paths relative to its root so `_members` reads alike.

    EVERY file, not only the volumes: the download's label tables ship beside them and are read through
    this same reader, and the volume consumer filters on `VOLUME_SUFFIX` itself. Listing only volumes
    here made an extracted corpus one that had pixels and no labels.
    """

    root: Path

    @override
    def names(self) -> list[str]:
        return [path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file()]

    @override
    def read(self, member: str) -> bytes:
        return (self.root / member).read_bytes()
