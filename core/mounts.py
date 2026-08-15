"""Finding a file under `/kaggle/input` without walking the competition data.

`Path("/kaggle/input").rglob(...)` was measured at **248 seconds** on 2026-08-14, before a line of ours
ran: it descends into the competition mount, which holds one directory per study and one per series
beneath that, and stats every DICOM on the way. The kernel scripts learned that and prune; the package
did not, and kept three unpruned walks — one of them inside the member-loading loop, so an entry with
twenty-two members paid for twenty-two of them.

`rglob` cannot express the fix. Its results can be filtered, but the filter runs after the walk has
already descended, so a skip-list applied to the paths it yields costs exactly as much as no skip-list
at all. Pruning has to happen inside the traversal, which is what `os.walk` allows and this class does.
"""

import os
from pathlib import Path

INPUT = Path("/kaggle/input")

SERIES = frozenset({"train_series", "test_series", "train_images", "test_images"})
"""Competition mounts holding one directory per study. Descending into one is the whole cost."""

DEPTH = 8
"""How deep under the root anything we look for has ever sat, plus room. The deepest is an encoder
mount — `models/metaresearch/dinov2/PyTorch/small/1` is six — and the bound is a guard rather than a
guess: it keeps a mount arriving under a name this does not know from restoring an unbounded walk."""


class Mounts:
    """The mounted inputs, searched by pruned walk and remembered once found."""

    def __init__(self, root: Path = INPUT, depth: int = DEPTH, skip: frozenset[str] = SERIES) -> None:
        self.root = root
        self.depth = depth
        self.skip = skip
        self.found: dict[str, list[Path]] = {}

    def every(self, name: str) -> list[Path]:
        """Every directory holding a file called `name`, walked ONCE per name and then remembered.

        Cached because the caller that hurt most asked per member: twenty-two members each searching
        for the same encoder config is one answer bought twenty-two times, and the price is the whole
        competition mount each time.
        """
        if name not in self.found:
            self.found[name] = list(self.walk(name))
        return self.found[name]

    def walk(self, name: str) -> list[Path]:
        """The pruned traversal itself: skip the study directories, and stop at the depth bound."""
        holding = []
        for parent, directories, files in os.walk(self.root):
            here = Path(parent)
            deep = len(here.relative_to(self.root).parts) >= self.depth
            directories[:] = [] if deep else [child for child in directories if child not in self.skip]
            if name in files:
                holding.append(here)
        return sorted(holding)

    def one(self, name: str, description: str) -> Path:
        """The single directory holding `name`, or a refusal that says which it found.

        Ambiguity is fatal rather than resolved by sort order: picking the first of two encoder mounts
        builds a member on weights it was not fitted on, which surfaces as a shape error if we are lucky
        and as a silently wrong model if we are not.
        """
        holding = self.every(name)
        if not holding:
            message = f"no {description} under {self.root} (looking for {name})"
            raise FileNotFoundError(message)
        if len(holding) > 1:
            message = f"{description} matches {len(holding)} mounts: {holding}"
            raise FileNotFoundError(message)
        return holding[0]
