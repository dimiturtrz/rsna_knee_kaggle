"""Which of our files a kernel actually loads, read from the imports rather than from the package list.

`kit_check` used to ask git for every commit touching `kneemri` and `core` since the kit was stamped,
which counts DISTANCE over the declared packages and not over the code a kernel runs. Measurement
modules land in `kneemri/` constantly — `band_coverage`, `crop_laterality`, an `Acquisition` property —
and each one refused a kit that was byte-identical everywhere a kernel reads. On 2026-08-17 the refusal
was three purely additive commits, proven spurious by hand on submission day. A check that is always
overridden is a check that will be overridden the one time it is right.

PARSED, NOT IMPORTED, for the same reason `kernel_script` parses: importing the entrypoint drags in
torch and the whole serving stack to answer a question about file paths. `ast` reads the import
statements — including the ones deferred inside functions, which `ast.walk` reaches and a top-level-only
reader would miss.
"""

import ast
from pathlib import Path

ENTRYPOINT = "kneemri/kaggle/kernel_run.py"
"""What a kernel script builds: everything after the wheels are installed hangs off `KernelRun`."""

KERNELS = "comp/kernels"
"""Where the kernel scripts live. Each is a root too: `KernelRun` is not the only thing a kernel imports
from the kit — the champion reaches `raptor_run`, `raptor_seat` and `raptor_model` from its own body, and
rooting at `kernel_run.py` alone left those outside every freshness check (knee-qx0dh.23)."""

PACKAGES = ("kneemri", "core")
"""The roots a kit ships. An import outside them is a dependency, versioned by the wheel, not by us."""


class KernelClosure:
    """The transitive import closure of the kernel entrypoint and every kernel script, confined to our packages."""

    def __init__(self, repo: Path, entrypoint: str = ENTRYPOINT, kernels: str = KERNELS) -> None:
        """Anchored at the repository root; every path this produces is relative to it, posix-separated."""
        self.repo = repo
        self.entrypoint = entrypoint
        self.kernels = kernels

    def module_file(self, module: str) -> str | None:
        """The repo-relative file a dotted module names, or None when it is not ours.

        A package name resolves to its `__init__.py`, since that is the file whose contents run when the
        package is imported and the file a commit to it would touch.
        """
        if module.split(".")[0] not in PACKAGES:
            return None
        stem = Path(*module.split("."))
        for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
            if (self.repo / candidate).is_file():
                return candidate.as_posix()
        return None

    def imported(self, path: str) -> list[str]:
        """Every dotted module named by an import anywhere in this file, ours or not.

        `from core.data.corpus import study` names two candidates — the package and the submodule — and
        which one it is cannot be told from the statement, so both are offered and `module_file`
        decides. Offering a name that resolves to nothing is free; missing a submodule is a file the
        closure would not carry.
        """
        tree = ast.parse((self.repo / path).read_text(encoding="utf-8"))
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
                modules.extend(f"{node.module}.{alias.name}" for alias in node.names)
        return modules

    def files(self) -> list[str]:
        """Every file of ours a kernel loads, sorted, the entrypoint included and the kernel scripts not.

        A script is a root but not a kit file: it ships as the kernel's `code_file`, never in the kit.
        """
        scripts = [path.relative_to(self.repo).as_posix() for path in self.repo.glob(f"{self.kernels}/*/*.py")]
        reached = {self.entrypoint}
        pending = [self.entrypoint, *scripts]
        while pending:
            for module in self.imported(pending.pop()):
                found = self.module_file(module)
                if found is not None and found not in reached:
                    reached.add(found)
                    pending.append(found)
        return sorted(reached)
