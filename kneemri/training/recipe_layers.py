"""A recipe as layers — TOML files in the order a line names them, then `--set` assignments over them.

Every layer is read into leaf names BEFORE anything types it, so a value from a file and the same value
from `--set` reach `TrainConfig.typed` as one text and cannot parse differently. The layers land as parser
defaults, which is what keeps a flag spelled on the line above all of them.

A file may name the files it stacks on with a top-level `extends = ["../base.toml", ...]`, resolved
against that file's own directory and expanded depth-first ahead of it, so a leaf recipe states only its
variable and the trunk it shares is written once. Its `[meta]` table says what the recipe is FOR — the
hypothesis, the control, the gate — and is never a training option.
"""

import os
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import cast

CONFIG = "--config"
SET = "--set"
EXTENDS = "extends"
META = "meta"


class RecipeStatus(StrEnum):
    """Where a recipe stands: waiting for the card, read, read and lost, or withdrawn unread."""

    QUEUED = "queued"
    DONE = "done"
    REFUTED = "refuted"
    RETIRED = "retired"


@dataclass(frozen=True)
class RecipeMeta:
    """What a recipe is for, merged down its `extends` chain with the leaf's own `[meta]` winning."""

    hypothesis: str = ""
    control: str = ""
    gate: str = ""
    results: str = ""
    status: RecipeStatus = RecipeStatus.QUEUED
    requires: tuple[str, ...] = ()
    bead: str = ""
    out: str = ""

    @classmethod
    def of(cls, table: Mapping[str, object]) -> "RecipeMeta":
        """The meta a merged `[meta]` table states; a key naming no field refuses, as a training key does."""
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(table) - known)
        if unknown:
            message = f"[{META}] names no recipe field: {', '.join(unknown)}"
            raise ValueError(message)
        stated = dict(table)
        stated["status"] = RecipeStatus(str(stated.get("status", RecipeStatus.QUEUED)))
        stated["requires"] = tuple(str(path) for path in cast("list[object]", stated.get("requires", [])))
        # pyrefly: ignore[bad-argument-type] - every key was checked against the fields above
        return cls(**stated)


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
        return {name: value for name, (value, _) in self.sources(addresses).items()}

    def sources(self, addresses: Mapping[str, str]) -> dict[str, tuple[object, str]]:
        """Each stated leaf's raw value with the layer that set it last — a file path, or `--set`."""
        stated: dict[str, tuple[object, str]] = {}
        for path, table in self._tables():
            options = {key: value for key, value in table.items() if key not in (EXTENDS, META)}
            named = self._named(dict(self._dotted(options, "")), addresses, str(path))
            stated |= {name: (value, str(path)) for name, value in named.items()}
        assigned = dict(self._assigned(assignment) for assignment in self.assignments)
        return stated | {name: (value, SET) for name, value in self._named(assigned, addresses, SET).items()}

    def meta(self) -> RecipeMeta:
        """The `[meta]` of every layer merged in stacking order, so the leaf's own fields win."""
        merged: dict[str, object] = {}
        for _, table in self._tables():
            merged |= cast("dict[str, object]", table.get(META, {}))
        return RecipeMeta.of(merged)

    def _tables(self) -> Iterator[tuple[Path, dict[str, object]]]:
        for config in self.configs:
            yield from self._expanded(config, ())

    @classmethod
    def _expanded(cls, path: Path, trail: tuple[Path, ...]) -> Iterator[tuple[Path, dict[str, object]]]:
        """This file's parents depth-first, then the file — refusing a file that reaches itself."""
        here = path.resolve()
        if here in trail:
            chain = " -> ".join(str(step) for step in (*trail, here))
            message = f"{path} extends itself: {chain}"
            raise ValueError(message)
        with path.open("rb") as handle:
            table = tomllib.load(handle)
        for parent in cast("list[str]", table.get(EXTENDS, [])):
            yield from cls._expanded(Path(os.path.normpath(path.parent / parent)), (*trail, here))
        yield path, table

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
