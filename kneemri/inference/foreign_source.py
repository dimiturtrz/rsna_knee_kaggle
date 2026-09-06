"""Lift named top-level symbols out of a foreign inference kernel, verbatim. Never retype their code.

A published Kaggle kernel is one flat script: the model classes, the decode, and the serving loop share a
module namespace with the top-level statements that load checkpoints and write a submission. Importing it
runs all of that, so the classes cannot simply be imported; and RETYPING them is how a port stops being a
replication — a transposed constant in a windowing percentile is indistinguishable from a bad model when
the number comes back low.

So the symbols are CUT OUT by `ast`, at their own source lines, and re-emitted as a module. What runs here
is byte-identical to what ran there, which is the only version of this that can answer "is their model
better" rather than "is my transcription of their model better".

THE SECTION IS THE UNIT, NOT THE FILE, AND THAT IS THE WHOLE POINT. A converted notebook is several
programs sharing one namespace, and they rebind the same constant to different values.
`bend_the_knee_to_dinov3_ensembled.py` binds `SLICE_BAND` THREE times: (0.2, 0.8) at :411 in the TRAINING
program, (0.12, 0.88) at :1622 in the inference program that serves the released folds, and (0.2, 0.8)
again at :2335 for a second model family — where `CROP_MM` also goes 130.0 -> 10_000.0, a crop of the
whole image instead of a 13 cm box around the knee. Neither the first binding nor the last is the one the
released checkpoints are served under. Resolving a name by exec-then-look-up silently takes the last, and
feeds their model an input their model has never seen; taking the first is the same mistake mirrored.

So a lift names the LINE RANGE of the program it is porting, and resolves each symbol to its first
binding INSIDE that range. That the ranges have to be stated is not overhead — it is the port declaring
which of the three programs in the file it claims to be replicating.

The emitted module is NOT importable on its own: their functions read module-level globals that the
kernel filled from a Kaggle input mount (`SERIES_ROOT`, `LABELS`, `models`, `CFG`, `DEV`, `AMP_PREF`).
Those are the caller's to inject, by name, after the module is built — see `ForeignModule.inject`.
"""

import ast
import logging
import types
from dataclasses import dataclass
from pathlib import Path

from core import cli

logger = logging.getLogger(__name__)

_DEFINITION = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign)


@dataclass(frozen=True)
class ForeignModule:
    """A module built from another script's source, plus the globals its author left for the caller."""

    name: str
    source: str

    def build(self) -> types.ModuleType:
        """Execute the lifted source in a fresh namespace.

        This runs another author's code, which is the point: the alternative is retyping it. What bounds
        the risk is the LIFT, not the exec — only the named symbols are present, and a lifted symbol is a
        definition, never a top-level statement that does something.
        """
        module = types.ModuleType(self.name)
        exec(compile(self.source, f"<{self.name}>", "exec"), module.__dict__)
        return module

    @staticmethod
    def inject(module: types.ModuleType, **globals_: object) -> None:
        """Bind the module-level names the original kernel filled from its Kaggle mount.

        Their functions resolve these at CALL time, so binding them after the module is built is the same
        thing the kernel did — not a workaround for the lift.
        """
        for key, value in globals_.items():
            setattr(module, key, value)


@dataclass(frozen=True)
class ForeignSource:
    """The symbols one foreign kernel can hand over, cut at their own source lines."""

    path: Path
    header: str = ""
    """Import lines the lifted symbols need; the kernel's own imports are not lifted with them."""
    scope: tuple[int, int] | None = None
    """The line range of the ONE program in this file being ported; unscoped reads the whole file."""

    def tree(self) -> ast.Module:
        """The kernel parsed once — every lookup below reads this."""
        return ast.parse(self.path.read_text(encoding="utf-8"))

    def _in_scope(self, node: ast.stmt) -> bool:
        """Whether a top-level statement belongs to the program this lift claims to be replicating."""
        if self.scope is None:
            return True
        low, high = self.scope
        return low <= node.lineno <= high

    @staticmethod
    def _bound(node: ast.stmt) -> list[str]:
        """The top-level names one statement binds; an unpacking assignment binds several."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return [node.name]
        if isinstance(node, ast.Assign):
            return [target.id for target in node.targets if isinstance(target, ast.Name)] + [
                element.id
                for target in node.targets
                if isinstance(target, ast.Tuple)
                for element in target.elts
                if isinstance(element, ast.Name)
            ]
        return []

    def _first_definitions(self, tree: ast.Module) -> dict[str, ast.stmt]:
        """Every top-level symbol at its FIRST binding — a rebinding later in the file is not this symbol."""
        found: dict[str, ast.stmt] = {}
        for node in tree.body:
            if not isinstance(node, _DEFINITION) or not self._in_scope(node):
                continue
            for name in self._bound(node):
                found.setdefault(name, node)
        return found

    def first_definitions(self) -> dict[str, ast.stmt]:
        """Where each symbol the kernel offers is cut from."""
        return self._first_definitions(self.tree())

    @staticmethod
    def _start(node: ast.stmt) -> int:
        """The first line of a definition INCLUDING its decorators.

        `ast` puts a decorated function's `lineno` on the `def`, not on the `@`, so cutting at `lineno`
        silently drops every decorator. That is not a formatting loss: their `_micro` is
        `@torch.no_grad()`, and lifted without it their serving loop builds a graph and dies on
        `Can't call numpy() on Tensor that requires grad` — the visible half of the failure. The
        invisible half is a decorator that changes VALUES rather than raising.
        """
        decorators = getattr(node, "decorator_list", [])
        return min([node.lineno, *[one.lineno for one in decorators]])

    def lift(self, names: list[str]) -> ForeignModule:
        """Cut the named symbols out in FILE order, so a definition still precedes its first use.

        Requesting a name the kernel does not define is a fault, not an empty result: it means the kernel
        moved under a port that still believes it knows the shape of it.
        """
        tree = self.tree()
        definitions = self._first_definitions(tree)
        if missing := [name for name in names if name not in definitions]:
            message = f"{self.path.name} defines none of {missing}"
            raise KeyError(message)
        wanted = {id(definitions[name]) for name in names}
        lines = self.path.read_text(encoding="utf-8").splitlines()
        blocks = ["\n".join(lines[self._start(node) - 1 : node.end_lineno]) for node in tree.body if id(node) in wanted]
        return ForeignModule(name=self.path.stem, source="\n\n".join([self.header, *blocks]))


def main() -> None:
    """Report what a kernel would hand over, and where each symbol is cut from."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--kernel", type=Path, required=True)
    command.parser.add_argument("--names", type=str, nargs="+", required=True)
    command.parser.add_argument("--scope", type=int, nargs=2, default=None, metavar=("FIRST", "LAST"))
    arguments = command.parsed()

    source = ForeignSource(arguments.kernel, scope=tuple(arguments.scope) if arguments.scope else None)
    definitions = source.first_definitions()
    command.report(
        [
            f"{name:<20} line {definitions[name].lineno}" if name in definitions else f"{name:<20} ABSENT"
            for name in arguments.names
        ]
    )


if __name__ == "__main__":
    main()
