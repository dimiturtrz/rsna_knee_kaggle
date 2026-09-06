"""Which members a kernel will ask for, read from the `KernelRun(...)` call in its script.

The sibling of `kernel_sources`: that module reads what a kernel DECLARES it will have attached, this
one reads what it will actually load, and `mount_check` is where the two meet. Both read the kernel
directory and neither runs anything, because the comparison has to be possible on a machine with no card
free and no Kaggle session — the whole point is to make it cheap enough to run before every submission.
"""

import ast
from pathlib import Path

KERNEL_RUN = "KernelRun"
"""The call every entry that runs OUR package makes, and the only place its member selection appears."""

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
