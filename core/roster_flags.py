"""`--roster` or `--candidates`: the two ways a pricing command is told which members to read.

Here rather than beside a roster reader because the commands that take them sit in both the evaluation
and the submission layer, and neither may import the other.
"""

import argparse
from pathlib import Path

from core.denominator import Denominator
from core.paths import DataRoot


class RosterFlags:
    """Declares the two flags and resolves what they name, so no command spells either out again."""

    EMPTY = "no checkpoint named by --roster or --candidates"

    @staticmethod
    def add(parser: argparse.ArgumentParser) -> None:
        """Declare both flags, a roster file winning over a candidate list when both are given."""
        parser.add_argument("--roster", type=Path, default=None, help="a roster file, one member per line")
        parser.add_argument(
            "--candidates", nargs="*", default=[], help="member names or checkpoint paths, instead of a roster"
        )

    @staticmethod
    def checkpoints(arguments: argparse.Namespace, root: DataRoot) -> list[Path]:
        """The checkpoints the parsed flags name, resolved under the root's models."""
        named = arguments.roster.read_text(encoding="utf-8").splitlines() if arguments.roster else arguments.candidates
        return root.checkpoints(list(named))

    @classmethod
    def required(cls, arguments: argparse.Namespace, root: DataRoot) -> list[Path]:
        """The named checkpoints, refusing a set that names none rather than pricing nothing."""
        paths = cls.checkpoints(arguments, root)
        Denominator(len(paths), cls.EMPTY).refuse_empty()
        return paths
