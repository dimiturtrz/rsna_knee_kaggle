"""Which members a kernel will ask for, read from the `KernelRun(...)` or `SeatedEntry(...)` call in its script.

The sibling of `kernel_sources`: that module reads what a kernel DECLARES it will have attached, this
one reads what it will actually load, and `mount_check` is where the two meet. Both read the kernel
directory and neither runs anything, because the comparison has to be possible on a machine with no card
free and no Kaggle session — the whole point is to make it cheap enough to run before every submission.
"""

import ast
from collections.abc import Iterator
from pathlib import Path
from typing import TypeGuard

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
