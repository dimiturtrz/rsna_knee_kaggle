"""Which members a kernel will ask for, read from the `KernelRun(...)` call in its script.

The sibling of `kernel_sources`: that module reads what a kernel DECLARES it will have attached, this
one reads what it will actually load, and `mount_check` is where the two meet. Both read the kernel
directory and neither runs anything, because the comparison has to be possible on a machine with no card
free and no Kaggle session — the whole point is to make it cheap enough to run before every submission.
"""

import ast
from pathlib import Path

from kneemri.kaggle.raptor_run import DEFAULT_WEIGHTS

KERNEL_RUN = "KernelRun"
"""The call every entry that runs OUR package makes, and the only place its member selection appears."""

RAPTOR_WEIGHTS = "weights"
"""`RaptorEntry.weights(...)`, the one call a foreign seat is loaded through, outside the `KernelRun` selection."""

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
        """The script's `KernelRun(...)`, or None when it never builds one.

        None is not "runs nothing": `kneemri-graft` loads whatever package the mount happens to hold via
        `MemberPackage.find`, and `kneemri-probe` loads no weights at all. Neither can be checked against
        our package, and saying so is the honest answer — a sweep that guessed would raise a false alarm
        on a kernel that is right.
        """
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == KERNEL_RUN:
                return node
        return None

    def member_names(self) -> list[str] | None:
        """The members the call names, or None when it takes whatever the package holds.

        None means EVERY member is a candidate, and the caller must read it that way: `KernelRun()` runs
        the package and `KernelRun(members=2)` runs a data-dependent two of it, so the mounts either
        entry needs are the package's whole set. Over-declaring costs mount seconds; under-declaring
        costs the slot.
        """
        call = self.call()
        if call is None:
            return None
        selected = next((keyword.value for keyword in call.keywords if keyword.arg == SELECTION), None)
        return None if selected is None else self._strings(selected)

    def foreign_seats(self) -> set[str]:
        """The foreign checkpoints the script loads, by the name `comp/board.jsonl` records them under.

        A raptor seat is blended in beside `KernelRun`, never through its `names`, so a check that reads
        only the selection sees seven of eight seats and calls a re-ship new (`tripleseatclean` against
        `tripleseat_v2`, ref 56232529). A call with no filename loads `DEFAULT_WEIGHTS`.
        """
        stems = {
            Path(filename).stem
            for call in ast.walk(self.tree)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == RAPTOR_WEIGHTS
            for filename in self._filenames(call)
        }
        return {BOARD_NAMES.get(stem, stem) for stem in stems}

    def _filenames(self, call: ast.Call) -> list[str]:
        """The checkpoint filenames one `weights(...)` call can open, through a variable or a loop over one."""
        given = next((keyword.value for keyword in call.keywords if keyword.arg == "filename"), None)
        argument = call.args[0] if call.args else given
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
