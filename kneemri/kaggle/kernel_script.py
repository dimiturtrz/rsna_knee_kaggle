"""Which members a kernel will ask for, read from the `KernelRun(...)` or `SeatedEntry(...)` call in its script.

The sibling of `kernel_sources`: that module reads what a kernel DECLARES it will have attached, this
one reads what it will actually load, and `mount_check` is where the two meet. Both read the kernel
directory and neither runs anything, because the comparison has to be possible on a machine with no card
free and no Kaggle session — the whole point is to make it cheap enough to run before every submission.
"""

import ast
import operator
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

from kneemri.kaggle.blend_shape import SHAPES, Fusion, Split
from kneemri.kaggle.raptor_run import DEFAULT_WEIGHTS

KERNEL_RUN = "KernelRun"
SEATED_ENTRY = "SeatedEntry"
SEAT_ENTRIES = frozenset({SEATED_ENTRY})
ENTRY_CALLS = frozenset({KERNEL_RUN}) | SEAT_ENTRIES
"""The calls an entry that runs OUR package makes, and the only places its member selection appears.
`SeatedEntry` builds its `KernelRun` inside the kit, so the kernel script shows only the outer call."""

RAPTOR_WEIGHTS = "weights"
"""`RaptorEntry.weights(...)`, the one call a foreign seat is loaded through, outside the `KernelRun` selection.
It is also `SeatedEntry`'s keyword for the same filename, which that class hands to `RaptorEntry.weights`."""

WEIGHTS_FILENAME = "filename"
"""`RaptorEntry.weights`' own keyword for the checkpoint filename, when it is not passed positionally."""

BOARD_NAMES = {"raptor_ft_coatnet_v4_full_swa": "raptor_v4_swa"}
"""A checkpoint stem the board records under another name; every other stem is its own board name."""

SELECTION = "names"
"""Its one keyword whose value resolves statically. `members=N` is `strongest(N)`, a ranking over
recorded holdouts that a parse cannot evaluate, so it reads as the whole package instead."""

FOREIGN_TABLES = {
    "nartaa_seat.csv": "nartaa384",
    "ours_sub.csv": "heliosli_s6",
    "reader_seat.csv": "goodpjw_convnext",
    "reader_ranks.csv": "goodpjw_convnext",
    "mavit_sub.csv": "hengck23_mavit",
}
"""The table a foreign producer writes inside a kernel, by the seat name `comp/board.jsonl` records its share under."""

FOREIGN_MARKERS = {
    **FOREIGN_TABLES,
    "nartaa_mirror.py": "nartaa384",
    "nartaa_probs.csv": "nartaa384",
    "nartaa_0945.py": "nartaa224",
    "helios_s6.py": "heliosli_s6",
    "cnxt_v0_fold0.pt": "goodpjw_convnext",
    "mavit_seat.py": "hengck23_mavit",
}
"""Every literal that says a foreign producer runs. A kernel carrying one whose blend this parse cannot resolve
is UNKNOWN, never all-ours: a per-finding share table or a nested wave has no scalar share to compare."""

PROBIT_TRANSFORM = "ndtri"
"""The inverse normal CDF a probit fusion maps each table's ranks through before the weighted sum."""

SHARE_PLACES = 4
"""Shares compare at four decimals, so a kernel's `2.0 / 3.0` meets the record's 0.6667."""

ARITHMETIC: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


@dataclass(frozen=True)
class Blend:
    """The part of an entry's identity outside our members: each foreign seat's share, and the fusion."""

    foreign: tuple[tuple[str, float], ...]
    fusion: Fusion

    @classmethod
    def of(cls, foreign: Mapping[str, float], fusion: str) -> "Blend":
        """A blend with its shares rounded and ordered, so two spellings of one blend compare equal."""
        rounded = sorted((seat, round(share, SHARE_PLACES)) for seat, share in foreign.items())
        return cls(tuple(rounded), Fusion(fusion))

    def described(self) -> str:
        """The blend as a reader writes it: `foreign {seat: share}, fusion rank`."""
        return f"foreign {dict(self.foreign)}, fusion {self.fusion}"

    def differences(self, other: "Blend") -> list[str]:
        """One `<axis> mine vs theirs` phrase per axis that differs; empty when the two are one entry."""
        axes = (("foreign", dict(self.foreign), dict(other.foreign)), ("fusion", str(self.fusion), str(other.fusion)))
        return [f"{axis} {mine} vs {theirs}" for axis, mine, theirs in axes if mine != theirs]


ALL_OURS = Blend((), Fusion.RANK)
"""An entry blending no foreign seat, which is also what a record row without the two fields says."""


class KernelScript:
    """Which members a kernel will ask for, read from the source rather than by running it.

    PARSED, NOT IMPORTED. A kernel script's module body finds a mounted kit and pip-installs wheels out
    of `/kaggle/input`; importing one to inspect it would either fail off-Kaggle or do something to this
    machine. `ast` reads the call the same way and does neither.
    """

    def __init__(self, script: Path) -> None:
        self.script = script
        self.tree = ast.parse(script.read_text(encoding="utf-8"))

    def call(self) -> ast.Call | None:
        """The script's `KernelRun(...)` or `SeatedEntry(...)`, or None when it builds neither.

        None is not "runs nothing": `kneemri-graft` loads whatever package the mount happens to hold via
        `MemberPackage.find`, and `kneemri-probe` loads no weights at all. Neither can be checked against
        our package, and saying so is the honest answer — a sweep that guessed would raise a false alarm
        on a kernel that is right.
        """
        for node in ast.walk(self.tree):
            if self._named(node, ENTRY_CALLS):
                return node
        return None

    def member_names(self) -> list[str] | None:
        """The members the call names, or None when it takes whatever the package holds.

        None means EVERY member is a candidate, and the caller must read it that way: `KernelRun()` runs
        the package and `KernelRun(members=2)` runs a data-dependent two of it, so the mounts either
        entry needs are the package's whole set. Over-declaring costs mount seconds; under-declaring
        costs the slot.
        """
        selections = [self._selection(call) for call in self._calls()]
        if not selections or None in selections:
            return None
        return list(dict.fromkeys(name for names in selections if names is not None for name in names))

    def _calls(self) -> list[ast.Call]:
        """Every entry call, in source order: `kneemri-twosoupmirror` runs one `KernelRun` per soup, and a
        check that read only the first saw one member of two (knee-5qv7k)."""
        calls = [node for node in ast.walk(self.tree) if self._named(node, ENTRY_CALLS)]
        return sorted(calls, key=lambda call: (call.lineno, call.col_offset))

    def _selection(self, call: ast.Call) -> list[str] | None:
        selected = self._keyword(call, SELECTION)
        return None if selected is None else self._strings(selected)

    def foreign_seats(self) -> set[str]:
        """The foreign checkpoints the script loads, by the name `comp/board.jsonl` records them under.

        A raptor seat is blended in beside `KernelRun`, never through its `names`, so a check that reads
        only the selection sees seven of eight seats and calls a re-ship new (`tripleseatclean` against
        `tripleseat_v2`, ref 56232529). A `SeatedEntry(...)` loads its `weights=` the same way, so it counts
        as one such call. A call with no filename loads `DEFAULT_WEIGHTS`.
        """
        stems = {
            Path(filename).stem for call, argument in self._seats() for filename in self._filenames(call, argument)
        }
        return {BOARD_NAMES.get(stem, stem) for stem in stems}

    def _seats(self) -> Iterator[tuple[ast.Call, ast.expr | None]]:
        """Every call that loads a foreign seat, with the filename expression it passes (None: the default)."""
        for node in ast.walk(self.tree):
            if self._named(node, SEAT_ENTRIES):
                yield node, self._keyword(node, RAPTOR_WEIGHTS)
            elif (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == RAPTOR_WEIGHTS
            ):
                yield node, node.args[0] if node.args else self._keyword(node, WEIGHTS_FILENAME)

    @staticmethod
    def _named(node: ast.AST, names: frozenset[str]) -> TypeGuard[ast.Call]:
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names

    @staticmethod
    def _keyword(call: ast.Call, name: str) -> ast.expr | None:
        return next((keyword.value for keyword in call.keywords if keyword.arg == name), None)

    def _filenames(self, call: ast.Call, argument: ast.expr | None) -> list[str]:
        """The checkpoint filenames one seat call can open, through a variable or a loop over one."""
        if argument is None:
            return [DEFAULT_WEIGHTS]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            return [argument.value]
        if not isinstance(argument, ast.Name):
            return []
        assigned = self._assigned(argument.id)
        if isinstance(assigned, ast.Constant) and isinstance(assigned.value, str):
            return [assigned.value]
        source = self._looped(call, argument.id) or assigned
        return [] if source is None else self._strings(source) or []

    def _looped(self, call: ast.Call, name: str) -> ast.expr | None:
        """What the comprehension or `for` loop AROUND `call` iterates over when it binds `name`."""
        loops = [
            loop.iter
            for node in ast.walk(self.tree)
            if isinstance(node, ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp | ast.For)
            and any(inner is call for inner in ast.walk(node))
            for loop in (node.generators if not isinstance(node, ast.For) else [node])
            if isinstance(loop.target, ast.Name) and loop.target.id == name
        ]
        return loops[-1] if loops else None

    def _strings(self, value: ast.expr) -> list[str] | None:
        """A literal sequence of names, following one level of variable, or None if it is not literal."""
        if isinstance(value, ast.Name):
            resolved = self._assigned(value.id)
            return None if resolved is None else self._strings(resolved)
        if not isinstance(value, ast.Tuple | ast.List):
            return None
        names = [
            element.value
            for element in value.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        ]
        return names if len(names) == len(value.elts) else None

    def _assigned(self, name: str) -> ast.expr | None:
        """The last value assigned to a bare name anywhere in the script."""
        assignments = [
            node.value
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        ]
        return assignments[-1] if assignments else None

    def marked_seats(self) -> set[str]:
        """The foreign seats the script's literals say it runs, whether or not its blend resolves."""
        return {
            FOREIGN_MARKERS[node.value]
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in FOREIGN_MARKERS
        }

    def blend(self) -> Blend | None:
        """The foreign shares and the fusion, or None when a foreign seat runs and no single blend prices it.

        A blend call counts only if it resolves every table and share AND covers every seat the script's
        literals mark, so a per-finding share table or a nested wave reads as unknown, never as a guess.
        """
        marked = self.marked_seats()
        if not marked:
            return ALL_OURS
        resolved = {self._resolved(split) for split in self._splits()}
        covering = {blend for blend in resolved if blend is not None and marked <= dict(blend.foreign).keys()}
        return next(iter(covering)) if len(covering) == 1 else None

    def _splits(self) -> list[Split]:
        probit = any(isinstance(node, ast.Name) and node.id == PROBIT_TRANSFORM for node in ast.walk(self.tree))
        calls = [node for node in ast.walk(self.tree) if isinstance(node, ast.Call)]
        splits = [shape.split(call, probit=probit) for call in calls for shape in SHAPES]
        return [split for split in splits if split is not None]

    def _resolved(self, split: Split) -> Blend | None:
        resolved = [self._tables(expression, frozenset()) for expression in split.tables]
        tables = [table for found in resolved if found is not None for table in found]
        weights = self._numbers(split.weights, frozenset())
        if weights is None or None in resolved:
            return None
        shares = [1.0 - sum(weights), *weights] if split.remainder else weights
        if len(shares) != len(tables) or None in tables:
            return None
        foreign: dict[str, float] = {}
        for table, share in zip(tables, shares, strict=True):
            seat = FOREIGN_TABLES.get(str(table))
            if seat is not None:
                foreign[seat] = foreign.get(seat, 0.0) + share
        return Blend.of(foreign, split.fusion)

    def _tables(self, expression: ast.expr, seen: frozenset[str]) -> list[str | None] | None:
        """The table filenames an expression holds, in order: a tuple, a comprehension over one, a slice of either."""
        if isinstance(expression, ast.Subscript):
            base = self._tables(expression.value, seen)
            return None if base is None else self._picked(base, expression.slice)
        if isinstance(expression, ast.Name):
            assigned = None if expression.id in seen else self._assigned(expression.id)
            return None if assigned is None else self._tables(assigned, seen | {expression.id})
        if isinstance(expression, ast.ListComp):
            return self._tables(expression.generators[0].iter, seen)
        if isinstance(expression, ast.Tuple | ast.List):
            return [self._table(element, seen) for element in expression.elts]
        return [self._table(expression, seen)]

    def _table(self, expression: ast.expr, seen: frozenset[str]) -> str | None:
        """One table's filename: a literal, the right side of `working / name`, a read of it, through a variable."""
        if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
            return expression.value
        if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Div):
            return self._table(expression.right, seen)
        if isinstance(expression, ast.Call) and expression.args:
            return self._table(expression.args[0], seen)
        if isinstance(expression, ast.Name) and expression.id not in seen:
            assigned = self._assigned(expression.id)
            return None if assigned is None else self._table(assigned, seen | {expression.id})
        return None

    @staticmethod
    def _picked(base: list[str | None], index: ast.expr) -> list[str | None] | None:
        """`base[i]` as a one-table list, or `base[i:j]` with literal bounds; None for anything computed."""
        position = KernelScript._integer(index)
        if position is not None:
            return base[position : position + 1] or None
        if not isinstance(index, ast.Slice) or index.step is not None:
            return None
        written = (index.lower, index.upper)
        lower, upper = (None if bound is None else KernelScript._integer(bound) for bound in written)
        computed = (bound is not None and value is None for bound, value in zip(written, (lower, upper), strict=True))
        return None if any(computed) else base[lower:upper]

    @staticmethod
    def _integer(node: ast.expr) -> int | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
        return None

    def _numbers(self, expression: ast.expr, seen: frozenset[str]) -> list[float] | None:
        """A literal sequence of shares, or one share as a list of one, through a variable."""
        if isinstance(expression, ast.Name) and expression.id not in seen:
            assigned = self._assigned(expression.id)
            return None if assigned is None else self._numbers(assigned, seen | {expression.id})
        elements = expression.elts if isinstance(expression, ast.Tuple | ast.List) else [expression]
        numbers = [self._number(element, seen) for element in elements]
        return None if None in numbers else [number for number in numbers if number is not None]

    def _number(self, expression: ast.expr, seen: frozenset[str]) -> float | None:
        """Constant arithmetic over literals and module constants such as `1.0 - OURS`; None for anything else."""
        if isinstance(expression, ast.Constant) and isinstance(expression.value, int | float):
            return None if isinstance(expression.value, bool) else float(expression.value)
        if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.USub):
            operand = self._number(expression.operand, seen)
            return None if operand is None else -operand
        if isinstance(expression, ast.Name) and expression.id not in seen:
            assigned = self._assigned(expression.id)
            return None if assigned is None else self._number(assigned, seen | {expression.id})
        if not isinstance(expression, ast.BinOp):
            return None
        applied = ARITHMETIC.get(type(expression.op))
        left, right = self._number(expression.left, seen), self._number(expression.right, seen)
        return None if applied is None or left is None or right is None else applied(left, right)
