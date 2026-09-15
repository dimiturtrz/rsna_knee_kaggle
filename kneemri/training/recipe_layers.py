"""A recipe as layers — TOML files in the order a line names them, then `--set` assignments over them.

Every layer is read into leaf names BEFORE anything types it, so a value from a file and the same value
from `--set` reach `TrainConfig.typed` as one text and cannot parse differently. The layers land as parser
defaults, which is what keeps a flag spelled on the line above all of them.
"""

import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

CONFIG = "--config"
SET = "--set"


@dataclass(frozen=True)
class RecipeLayers:
    """The files and assignments a command line stacks over the tree's defaults, later ones winning."""

    configs: tuple[Path, ...] = ()
    assignments: tuple[str, ...] = ()

    @classmethod
    def of(cls, argv: list[str]) -> "RecipeLayers":
        """Every `--config` and `--set` on the line, in order — read off the tokens, as argparse keeps one."""
        return cls(tuple(Path(text) for text in cls._after(argv, CONFIG)), tuple(cls._after(argv, SET)))

    @staticmethod
    def _after(argv: list[str], flag: str) -> Iterator[str]:
        return (argv[index + 1] for index, token in enumerate(argv[:-1]) if token == flag)

    def values(self, addresses: Mapping[str, str]) -> dict[str, object]:
        """Each stated leaf's raw value by leaf name, from the dotted `addresses` the tree declares.

        A path naming no leaf REFUSES rather than drops: a misspelt `stochastic_dept = 0.05`, or a knob
        filed under the wrong node, read as nothing is a run under a recipe nobody wrote.
        """
        stated: dict[str, object] = {}
        for config in self.configs:
            with config.open("rb") as handle:
                stated |= self._named(dict(self._dotted(tomllib.load(handle), "")), addresses, str(config))
        assigned = dict(self._assigned(assignment) for assignment in self.assignments)
        return stated | self._named(assigned, addresses, SET)

    @staticmethod
    def _named(table: Mapping[str, object], addresses: Mapping[str, str], source: str) -> dict[str, object]:
        unknown = sorted(set(table) - set(addresses))
        if unknown:
            message = f"{source} names no training option: {', '.join(unknown)}"
            raise ValueError(message)
        return {addresses[path]: value for path, value in table.items()}

    @staticmethod
    def _dotted(table: Mapping[str, object], prefix: str) -> Iterator[tuple[str, object]]:
        for key, value in table.items():
            if isinstance(value, dict):
                yield from RecipeLayers._dotted(cast("Mapping[str, object]", value), f"{prefix}{key}.")
            else:
                yield f"{prefix}{key}", value

    @staticmethod
    def _assigned(assignment: str) -> tuple[str, object]:
        path, _, text = assignment.partition("=")
        try:
            value = tomllib.loads(f"value = {text}")["value"]
        except tomllib.TOMLDecodeError:
            return path, text
        return path, value
