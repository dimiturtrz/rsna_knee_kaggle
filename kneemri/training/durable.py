"""A write that survives the machine, not merely the process.

Temp-then-rename is atomic against a dying PROCESS and not against a dying MACHINE. Without an fsync
the filesystem may journal the file's size while its data blocks are still in cache: an OS restart then
leaves a file of exactly the right length whose tail is zeros and which carries no zip central
directory, so `torch.load` raises `PytorchStreamReader failed reading zip archive` on a checkpoint that
every other signal — size, header, modification time — says is intact.

Its own module because both halves of a killed run's recovery need it and they may not import each
other: the weights are written by `Trainer`, the trajectory beside them by `ResumeState`.
"""

import os
from pathlib import Path

import torch


class Durable:
    """A file whose bytes are on the platter before its name appears."""

    @staticmethod
    def save(payload: dict[str, object], path: Path) -> None:
        """Write `payload` to `path`, flushed then renamed, so a reader sees all of it or none of it."""
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
