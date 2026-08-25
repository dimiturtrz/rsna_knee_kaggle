"""The four statements every `python -m` entrypoint in this repository opens with, written once.

MEASURED over `kneemri/` and `core/` on 2026-08-25, 163 modules: `ArgumentParser(description=__doc__)`
appears in 55, `logging.basicConfig(level=logging.INFO, format="%(message)s")` in 60, a `--paths`
argument defaulting to `paths.yaml` in 45, and `DataRoot.from_config(arguments.paths)` in 45. Three
entrypoints read in full — `band_coverage`, `label_stability`, `paired_floor` — open with the same five
statements in the same order before a single module-specific line appears, and twenty-four of them close
with the same `for line in ...: logger.info(line)`.

That is the repository's only CLI contract, and it was written out four dozen times instead of once.

DELIBERATELY NOT A FRAMEWORK. There is no decorator, no registry, and no declarative argument DSL. A
module still reaches for `add_argument` with its own flags and still owns its `main`; what it stops doing
is re-deciding the log format and re-spelling where paths.yaml lives. The parser is exposed as an
attribute rather than wrapped, so anything argparse can express stays available without this class
growing a passthrough for it.

    def main() -> None:
        \"\"\"What this command answers.\"\"\"
        command = cli.Command(__doc__, logger)
        command.parser.add_argument("--sample", type=int, default=SAMPLE)
        arguments = command.parsed()
        root = command.rooted(arguments)
        ...
        command.report(BandCoverage(band).report(measured, len(chosen), len(identifiers)))

The bar it has to clear is that an entrypoint using it reads SHORTER and no less obviously than the six
lines it replaces. If hiding `parse_args` behind a method would break that, it stays visible — which is
why `parsed()` returns the namespace to the caller rather than keeping it.
"""

import argparse
import logging
from collections.abc import Iterable
from pathlib import Path

from core.paths import DataRoot

FORMAT = "%(message)s"
"""Bare message, no level or timestamp. These commands print reports for a person, not a service log."""

PATHS = Path("paths.yaml")
"""The gitignored machine-local config. Every command takes `--paths` so a runner can redirect it."""


class Command:
    """One `python -m` entrypoint's opening: its parser, its logging, its data root, its report sink.

    Constructed with the module's own docstring and the module's own logger. The docstring, because a
    description kept beside the module is a second thing to keep true and it drifts — the same argument
    `kneemri.commands` makes for generating the command index. The logger, because a report line should
    name the command that produced it: routing twenty-four commands' output through `core.cli` would
    relabel every line in every one of them.
    """

    def __init__(self, doc: str | None, log: logging.Logger) -> None:
        self.log = log
        self.parser = argparse.ArgumentParser(description=doc)
        self.parser.add_argument("--paths", type=Path, default=PATHS, help="machine-local data locations")

    def parsed(self) -> argparse.Namespace:
        """Read the command line, and send this process's logging to the console in the report format.

        The two together because every entrypoint here does both, in this order, and a command that
        parsed without configuring logging would print nothing at all — a failure that looks like an
        empty result rather than like a missing line.
        """
        arguments = self.parser.parse_args()
        logging.basicConfig(level=logging.INFO, format=FORMAT)
        return arguments

    def rooted(self, arguments: argparse.Namespace) -> DataRoot:
        """The data root the parsed `--paths` names."""
        return DataRoot.from_config(arguments.paths)

    def report(self, lines: Iterable[str]) -> None:
        """Print a report a line at a time, through the calling module's logger."""
        for line in lines:
            self.log.info(line)
